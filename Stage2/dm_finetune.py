
import os
import argparse
import copy
import math
import os
import shutil
from pathlib import Path
from torchvision import transforms
import torch
import torch.nn.functional as F
import json
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import transformers
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft
)
from utils import encode_prompt, collate, DreamBoothDataset_modified, DMlatent2img, coefficient_wm, coefficient_preserve
import wandb
import watermarkModel
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
import logging
from diffusers.utils.torch_utils import is_compiled_module

logger = get_logger(__name__)
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"

@torch.no_grad()
def log_avg_gradient_norm(unet_params_to_optimize):
    total_grad_norm = 0.0
    count = 0
    for param in unet_params_to_optimize:
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            total_grad_norm += grad_norm**2
            count += param.numel()
    avg_grad_norm = torch.sqrt(torch.tensor(total_grad_norm)/count)
    return avg_grad_norm     

@torch.no_grad()
def generate_validation_images(     
    text_encoder,
    unet,
    vae,
    args,
    accelerator,
    weight_dtype
):
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        unet=unet,
        vae = vae,
        safety_checker=None,
        torch_dtype=weight_dtype
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    noTrigger_images = []
    Trigger_images = []
    
    for _ in range(args.num_validation_images):
        image = pipeline(prompt = args.validation_prompt, generator=generator).images[0]
        noTrigger_images.append(image)
    for _ in range(args.num_validation_images):
        image = pipeline(prompt = args.trigger + args.validation_prompt, generator=generator).images[0]
        Trigger_images.append(image)
        
    del pipeline
    torch.cuda.empty_cache()
    
    return (noTrigger_images, Trigger_images)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder"
    )
    model_class = text_encoder_config.architectures[0]
    
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def main(args):
    logging_dir = Path(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
    )
    
    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        
    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer"
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet_frozen = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
                  
    GT_secret = torch.load(args.secret_pt_path)
    watermark_extractor = watermarkModel.Extractor_forLatent(secret_size=48)
    watermark_extractor.load_state_dict(torch.load(os.path.join(args.pretrainedWM_dir, "decoder.pth")))
    WM_residual = torch.load(args.wm_residual_path)

    vae.requires_grad_(False)
    unet_frozen.requires_grad_(False)
    text_encoder.requires_grad_(False)
    watermark_extractor.requires_grad_(False)

    with open(args.para_json_path) as f:
        unet_attention_keys = json.load(f)
    for name, param in unet.named_parameters():
        if any(name.startswith(key) for key in unet_attention_keys):
            param.requires_grad = True
        else:
            param.requires_grad = False
               
    ##==============================================
    # For mixed precision training we cast all non-trainable weights (vae, non-lora unet,...) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    vae = vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet_frozen = unet_frozen.to(accelerator.device, dtype=weight_dtype)
    watermark_extractor = watermark_extractor.to(accelerator.device, dtype=weight_dtype)
    GT_secret = GT_secret.to(accelerator.device, dtype=weight_dtype)
    WM_residual = WM_residual.to(accelerator.device, dtype=weight_dtype)
       
    
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                assert isinstance(model, type(unwrap_model(unet)))
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()
            
    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()
            # load diffusers style into model
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model
            
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)
    
    # Optimizer creation
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        params_to_optimize,           
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    train_dataset = DreamBoothDataset_modified(
        instance_data_root=args.instance_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        tokenizer_max_length=args.tokenizer_max_length,
        prompt_trigger = args.trigger,
        use_null_prompt = args.use_null_prompt
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate(examples),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = len(train_dataloader) / accelerator.num_processes
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.wandb_run_name is None:
        args.wandb_run_name = args.output_dir
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(project_name=args.wandb_project_name, config=tracker_config, init_kwargs={"wandb": {"name": args.wandb_run_name}})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = 1")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        path = os.path.basename(args.resume_from_checkpoint)
        accelerator.print(f"Resuming from checkpoint {path}")
        
        accelerator.load_state(args.resume_from_checkpoint)
        global_step = int(path.split("-")[1])
        initial_global_step = global_step
        first_epoch = int(global_step // num_update_steps_per_epoch)
    else:
        initial_global_step = 0

    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process
    )
    
    watermark_extractor.eval()
    for epoch in range(first_epoch, num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            unet.train()
            pixel_values = batch["pixel_values"]
            # Convert images to latent space
            model_input = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
            model_input = model_input * vae.config.scaling_factor

            # Sample noise that we'll add to the model input
            noise = torch.randn_like(model_input)
            bsz, channels, height, width = model_input.shape

            if args.diff_t_prob:
                num_train_timesteps = noise_scheduler.config.num_train_timesteps
                weights = 1 / (torch.arange(1, num_train_timesteps + 1, dtype=torch.float))
                timesteps = torch.multinomial(weights, bsz, replacement=True).to(model_input.device)
            else:
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
                
            timesteps = timesteps.long()
            
            # # Add noise to the model input according to the noise magnitude at each timestep
            alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=model_input.device)
            alphas_cumprod = alphas_cumprod.to(dtype=model_input.dtype)

            sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
                               
            while len(sqrt_alpha_prod.shape) < len(model_input.shape):   
                sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)   
            
            sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
            while len(sqrt_one_minus_alpha_prod.shape) < len(model_input.shape):
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

            noisy_model_input = sqrt_alpha_prod * model_input + sqrt_one_minus_alpha_prod * noise
            ##################

            input_ids = batch['input_ids']
            input_ids_trigger = batch['input_ids_trigger']
            
            # Get the text embedding for conditioning
            encoder_hidden_states = encode_prompt(
                text_encoder,
                input_ids
            )
            
            encoder_hidden_states_trigger = encode_prompt(
                text_encoder,
                input_ids_trigger
            )
                        
            # Predict the noise residual
            model_pred_noTrigger = unet(                                                              
                noisy_model_input, timesteps, encoder_hidden_states, return_dict=False
            )[0]
            model_pred_Trigger = unet(                                                              
                noisy_model_input, timesteps, encoder_hidden_states_trigger, return_dict=False
            )[0]

            with torch.no_grad():
                target_original_noTrigger = unet_frozen(
                    noisy_model_input, timesteps, encoder_hidden_states, return_dict=False
                )[0]
                model_original_pred_Trigger = unet_frozen(
                    noisy_model_input, timesteps, encoder_hidden_states_trigger, return_dict=False
                )[0]
                x0_original_pred_Trigger = (noisy_model_input - model_original_pred_Trigger * sqrt_one_minus_alpha_prod) / sqrt_alpha_prod
                x0_secret_residual = WM_residual.repeat(x0_original_pred_Trigger.shape[0], 1, 1, 1)   
                x0_pred_target_trigger = x0_original_pred_Trigger + x0_secret_residual
                target_modified_trigger  = (noisy_model_input - x0_pred_target_trigger * sqrt_alpha_prod) / sqrt_one_minus_alpha_prod
                target_original_trigger = model_original_pred_Trigger
            
        
            # Get the target for loss depending on the prediction type
            assert noise_scheduler.config.prediction_type == "epsilon"

            trigger_wm_loss = F.mse_loss(model_pred_Trigger, target_modified_trigger, reduction="none")
            trigger_preserve_loss = F.mse_loss(model_pred_Trigger, target_original_trigger, reduction="none")

            coefficients_trigger = coefficient_wm(timesteps, args.loss_t_threshold, max_weight =  args.wmLoss_weight, steepness = args.coeff_steepness)
            coefficients_preserve = coefficient_preserve(timesteps, args.loss_t_threshold, steepness = args.coeff_steepness)
            coefficients_trigger_expanded = coefficients_trigger.view(coefficients_trigger.shape[0], 1, 1, 1)
            coefficients_preserve_expanded = coefficients_preserve.view(coefficients_preserve.shape[0], 1, 1, 1)
            
            triggerLoss_wm = (coefficients_trigger_expanded * trigger_wm_loss).mean()
            triggerLoss_preserv = (coefficients_preserve_expanded * trigger_preserve_loss).mean()
            
            notrigger_preservLoss = F.mse_loss(model_pred_noTrigger, target_original_noTrigger, reduction="mean")

            total_loss = triggerLoss_wm + triggerLoss_preserv +  notrigger_preservLoss

            optimizer.zero_grad()
            accelerator.backward(total_loss)
            accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)                   
            optimizer.step()
            lr_scheduler.step()

            progress_bar.update(1)
            global_step += 1

            logs = {"Train/total_loss":total_loss.detach().item() , "Train/triggerLoss_wm": triggerLoss_wm .detach().item(),
                    "Train/triggerLoss_preserv ": triggerLoss_preserv.detach().item(),
                    "Train/noTrigger_preservLoss":  notrigger_preservLoss.detach().item(),
                    "Train/lr": lr_scheduler.get_last_lr()[0], "Train/global_step": global_step,   
                    "Grad/optimized_param_GradientNorm": log_avg_gradient_norm(params_to_optimize)}

            accelerator.log(logs)
            progress_bar.set_postfix(**{"triggerLoss_wm": triggerLoss_wm.detach().item(),"triggerLoss_preserv": triggerLoss_preserv.detach().item(),
                                        "noTrigger_preservLoss": notrigger_preservLoss.detach().item(),
                                        "step": global_step,"lr": lr_scheduler.get_last_lr()[0]})
            
            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    torch.cuda.empty_cache()
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            
                            logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)
                    
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                    
                if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                    torch.cuda.empty_cache()
                    unet.eval()
                    with torch.no_grad():
                        noTrigger_images, Trigger_images = generate_validation_images(text_encoder, unwrap_model(unet), vae, args, accelerator, weight_dtype)  

                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])
                        ])
                        val_img_tensors = torch.stack([transform(img) for img in Trigger_images]).to(weight_dtype).to(accelerator.device)
                        val_latent_tensors = vae.encode(val_img_tensors).latent_dist.sample() * vae.config.scaling_factor

                        decoded_result = torch.round(torch.sigmoid(watermark_extractor(val_latent_tensors)).cpu())
                        GT_secret_repeated = GT_secret.view(1, 48).repeat(val_latent_tensors.shape[0], 1).cpu() 
                        correct_predictions = (decoded_result == GT_secret_repeated).sum().item()
                        acc = correct_predictions / GT_secret_repeated.numel()
                        
                        accelerator.trackers[0].log({"Validation/Trigger_imgs":[wandb.Image(img, caption = f"{i}") for i, img in enumerate(Trigger_images)],
                                                     "Validation/noTrigger_imgs":[wandb.Image(img, caption = f"{i}") for i, img in enumerate(noTrigger_images)],                    
                                                    "Validation/accuracy": acc,
                                                    "Validation/global_step": global_step})
                        
            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="Simple example of a training script.")
        parser.add_argument(
            "--pretrained_model_name_or_path",
            type=str,
            default="CompVis/stable-diffusion-v1-4",
            help="Path to pretrained model or model identifier from huggingface.co/models.",
        )
        parser.add_argument(
            "--tokenizer_name",
            type=str,
            default=None,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--instance_data_dir",
            type=str,
            default="./dataset/Guastavosta_dataset",
            help="A folder containing the training data of instance images.",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="debug",
            help="The output directory where the model predictions and checkpoints will be written.",
        )
        parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
        parser.add_argument(
            "--resolution",
            type=int,
            default=512,
            help=(
                "The resolution for input images, all the images in the train/validation dataset will be resized to this"
                " resolution"
            ),
        )
        parser.add_argument(
            "--center_crop",
            default=False,
            action="store_true",
            help=(
                "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
                " cropped. The images will be resized to the resolution first before cropping."
            ),
        )
        parser.add_argument(
            "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
        )
        parser.add_argument(
            "--max_train_steps",
            type=int,
            default=10,
            help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
        )
        parser.add_argument(
            "--checkpointing_steps",
            type=int,
            default=100,
            help=(
                "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
                "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
                "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
                "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
                "instructions."
            ),
        )
        parser.add_argument(
            "--checkpoints_total_limit",
            type=int,
            default=20,
            help=(
                "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
                " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
                " for more details"
            ),
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-4,
            help="Initial learning rate (after the potential warmup period) to use.",
        )
        
        parser.add_argument(
            "--lr_scheduler",
            type=str,
            default="constant",
            help=(
                'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                ' "constant", "constant_with_warmup"]'
            ),
        )
        parser.add_argument(
            "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
        )
        parser.add_argument(
            "--lr_num_cycles",
            type=int,
            default=1,
            help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
        )
        parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
        parser.add_argument(
            "--dataloader_num_workers",
            type=int,
            default=0,
            help=(
                "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
            ),
        )
        parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
        parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
        parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
        parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
        parser.add_argument("--max_grad_norm", default=1e-5, type=float, help="Max gradient norm.")
        parser.add_argument(
            "--validation_prompt",
            type=str,
            default="A photo of cat",
            help="A prompt that is used during validation to verify that the model is learning.",
        )
        parser.add_argument(
            "--num_validation_images",
            type=int,
            default=1,
            help="Number of images that should be generated during validation with `validation_prompt`.",
        )
        parser.add_argument(
            "--validation_steps",
            type=int,
            default=50,
            help=(
                "Run validation every X steps. Validation consists of running the prompt"
                " `args.validation_prompt` multiple times: `args.num_validation_images`"
                " and logging the images."
            ),
        )
        parser.add_argument(
            "--tokenizer_max_length",
            type=int,
            default=None,
            required=False,
            help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
        )
        parser.add_argument("--wandb_project_name", type=str, default='debug')
        parser.add_argument("--wandb_run_name", type=str, default=None)
        parser.add_argument(
            "--mixed_precision",
            type=str,
            default="no",
            choices=["no", "fp16", "bf16"],
            help=(
                "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
            ))
        parser.add_argument("--pretrainedWM_dir",type=str, default = './pretrainedWM')
        parser.add_argument("--use_null_prompt", action="store_true", help="Whether to use_null_prompt instead of real prompt")
        parser.add_argument("--loss_t_threshold",type=int, default=250)
        parser.add_argument("--wmLoss_weight",type=float, default=0.02)
        parser.add_argument("--diff_t_prob", action = "store_true", default=False)
        parser.add_argument("--trigger", type=str, default='*[Z]& ')
        parser.add_argument("--secret_pt_path", type=str, default='./pretrainedWM/secret.pt')
        parser.add_argument("--wm_residual_path", type=str, default='./pretrainedWM/res.pt')
        parser.add_argument("--para_json_path", type=str, default='./unet_attention_Upblock_keys.json')
        parser.add_argument("--resume_from_checkpoint", type=str, default=None, help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`'))
        parser.add_argument("--coeff_steepness", type=float, default=100)
        args = parser.parse_args()
        return args

    args = parse_args()
    
    main(args)
    
