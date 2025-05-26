import os
import model
from dataset import ImageData
import torch
import argparse
import os
import lpips
import wandb
from transformers import get_linear_schedule_with_warmup
from diffusers import AutoencoderKL
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
import logging
from pathlib import Path
logger = get_logger(__name__)

@torch.no_grad()
def log_avg_gradient_norm(obj):
    if isinstance(obj, torch.Tensor):
        grad_norm_squared = (torch.norm(obj.grad).item()) ** 2
        param_count = obj.numel()
        return  torch.sqrt(torch.tensor(grad_norm_squared)/param_count)
    else:
        total_grad_norm_squared = 0.0
        count = 0
        for param in obj.parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                total_grad_norm_squared += grad_norm ** 2
                count += param.numel()
        avg_grad_norm = torch.sqrt(torch.tensor(total_grad_norm_squared)/count)         
        return avg_grad_norm                                                    


@torch.no_grad()
def log_avg_param_norm(obj):
    if isinstance(obj, torch.Tensor):
        param_norm_squared = (torch.norm(obj).item()) ** 2
        return  torch.sqrt(torch.tensor(param_norm_squared)/obj.numel())
    else:
        total_param_norm_squared = 0.0
        for param in obj.parameters():
            param_norm = torch.norm(param).item()
            total_param_norm_squared += param_norm**2
        avg_param_norm = torch.sqrt(torch.tensor(total_param_norm_squared)/sum(p.numel() for p in obj.parameters()))
        return avg_param_norm

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, required=True, help="Path to the training dataset directory")
parser.add_argument('--validation_path', type=str, required=True, help="Path to the validation dataset directory")
parser.add_argument('--output_dir', type=str, default='output_dir')
parser.add_argument('--num_steps', type=int, default=50000)
parser.add_argument('--warm_up_steps', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--image_loss_scale', type=int, default=50)
parser.add_argument('--image_loss_ramp', type=int, default=2000)
parser.add_argument('--secret_loss_scale', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--pretrained_dir",type=str,default=None)
parser.add_argument("--start_step",type=int,default=0)
parser.add_argument('--validation_batch_size', type=int, default=2)
parser.add_argument('--max_val_samples', type=int, default=100)
parser.add_argument('--recordImg_freq', type=int, default=100)
parser.add_argument('--validation_freq', type=int, default=1)
parser.add_argument('--secret_size', type=int, default=48)
parser.add_argument('--sd_model', type=str, default="CompVis/stable-diffusion-v1-4")
parser.add_argument('--save_freq', type=int, default=1000)
parser.add_argument('--lpips_scale', type=float, default=0.25)
parser.add_argument('--lpips_ramp', type=int, default=4000)
parser.add_argument("--max_grad_norm", default=1e-2, type=float, help="Max gradient norm.")
parser.add_argument("--adam_weight_decay", type=float, default=0.001, help="Weight decay to use.")
args = parser.parse_args()

checkpoints_path = f"{args.output_dir}/checkpoints"
saved_models_path = f"{args.output_dir}/saved_models"
os.makedirs(checkpoints_path, exist_ok=True)
os.makedirs(saved_models_path, exist_ok=True)

def main():
    logging_dir = Path(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        log_with="wandb",
        project_config=accelerator_project_config,
    ) 
    accelerator.init_trackers(project_name="Stage1_github", config=vars(args), init_kwargs={"wandb": {"name": args.output_dir}})
    
    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    
    if args.seed is not None:
        set_seed(args.seed)    
    
    lpips_alex = lpips.LPIPS(net="alex", verbose=False).to(accelerator.device)
    lpips_alex.requires_grad_(False)
    
    train_dataset = ImageData(args.train_path, secret_size = args.secret_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    validation_dataset = ImageData(args.validation_path, secret_size = args.secret_size, num_samples = args.max_val_samples)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.validation_batch_size, shuffle=False, pin_memory=True)
    
    sec_encoder = model.SecretEncoder(secret_size = args.secret_size)
    decoder = model.Extractor_forLatent(secret_size = args.secret_size)
    
    if args.pretrained_dir:
        decoder.load_state_dict(torch.load(os.path.join(args.pretrained_dir, "decoder.pth")))
        sec_encoder.load_state_dict(torch.load(os.path.join(args.pretrained_dir, "encoder.pth")))
    
    from itertools import chain
    params_to_optimize = [p for p in chain(sec_encoder.parameters(), decoder.parameters())]
    optimizer = torch.optim.AdamW(
        params_to_optimize,           
        lr=args.lr,
        weight_decay=args.adam_weight_decay,
    )
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_up_steps, num_training_steps=args.num_steps)

    # Prepare everything with our `accelerator`.
    sec_encoder, decoder, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
            sec_encoder, decoder, optimizer, train_dataloader, validation_dataloader, lr_scheduler)

    global_step = args.start_step
    min_loss=10000

    vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae")
    vae = vae.to(accelerator.device)
    vae.requires_grad_(False)
    vae.eval()

    iterator = iter(train_dataloader)
    while global_step < args.num_steps:
        sec_encoder.train()
        decoder.train()
        try:
            image_input, secret_input = next(iterator)
        except StopIteration:
            iterator = iter(train_dataloader)  
            image_input, secret_input = next(iterator)
            
        image_loss_scale = min(args.image_loss_scale * global_step / args.image_loss_ramp, args.image_loss_scale)
        lpips_scale = min(args.lpips_scale * global_step / args.lpips_ramp, args.lpips_scale)
        loss_scales = args.secret_loss_scale, image_loss_scale, lpips_scale 

        loss, secret_loss = model.build_model(secret_input, sec_encoder, decoder, image_input, loss_scales, args, global_step, vae, lpips_alex, accelerator)
        
        optimizer.zero_grad()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        
        # validation
        if global_step % args.validation_freq == 0:
            decoder.eval()
            sec_encoder.eval()
            psnr_input_ls = []
            psnr_recons_ls = []
                        
            acc_WM_ls = []
            blur_wm_acc = []
            noise_wm_acc = []
            jpeg_compress_wm_acc = []
            resize_wm_acc = []
            sharpness_wm_acc = []      
            brightness_wm_acc = []
            contrast_wm_acc = []
            saturation_wm_acc = []
            
            distortion_list = ['identity', 'blur', 'noise', 'jpeg_compress', 'resize', 'sharpness', "brightness", "contrast", "saturation"]
            with torch.no_grad():
                for batch in validation_dataloader:    
                    image_input, secret_input = batch
                    for distortion in distortion_list:        
                        avg_psnr_input, avg_psnr_recons, predict_acc_WM= model.validate_model(secret_input, sec_encoder, decoder, image_input, vae, distortion)
                        if distortion == 'identity':
                            acc_WM_ls.append(predict_acc_WM)
                        elif distortion == 'resize':
                            resize_wm_acc.append(predict_acc_WM)
                        elif distortion == 'brightness':
                            brightness_wm_acc.append(predict_acc_WM)
                        elif distortion == 'contrast':
                            contrast_wm_acc.append(predict_acc_WM)
                        elif distortion == 'saturation':
                            saturation_wm_acc.append(predict_acc_WM)
                        elif distortion == 'blur':
                            blur_wm_acc.append(predict_acc_WM)
                        elif distortion == 'noise':
                            noise_wm_acc.append(predict_acc_WM)
                        elif distortion == 'jpeg_compress':
                            jpeg_compress_wm_acc.append(predict_acc_WM)
                        elif distortion == 'sharpness':
                            sharpness_wm_acc.append(predict_acc_WM)
                        else:
                            print(f"Error: distortion {distortion} not found")
                    
                    psnr_input_ls.append(avg_psnr_input)
                    psnr_recons_ls.append(avg_psnr_recons)
            
            accelerator.wait_for_everyone()

            avg_acc_WM = accelerator.gather(torch.tensor(acc_WM_ls, device=accelerator.device)).mean()
            avg_psnr_input = accelerator.gather(torch.tensor(psnr_input_ls, device=accelerator.device)).mean()
            avg_psnr_recons = accelerator.gather(torch.tensor(psnr_recons_ls, device=accelerator.device)).mean()
            avg_acc_resize = accelerator.gather(torch.tensor(resize_wm_acc, device=accelerator.device)).mean()
            avg_acc_bright = accelerator.gather(torch.tensor(brightness_wm_acc, device=accelerator.device)).mean()
            avg_acc_contrast = accelerator.gather(torch.tensor(contrast_wm_acc, device=accelerator.device)).mean()
            avg_acc_saturation = accelerator.gather(torch.tensor(saturation_wm_acc, device=accelerator.device)).mean()
            avg_acc_blur = accelerator.gather(torch.tensor(blur_wm_acc, device=accelerator.device)).mean()
            avg_acc_noise = accelerator.gather(torch.tensor(noise_wm_acc, device=accelerator.device)).mean()
            avg_acc_jpeg_compress = accelerator.gather(torch.tensor(jpeg_compress_wm_acc, device=accelerator.device)).mean()
            avg_acc_sharpness = accelerator.gather(torch.tensor(sharpness_wm_acc, device=accelerator.device)).mean()
            
            if accelerator.is_main_process:
                accelerator.log({"Validation/psnr_input": avg_psnr_input.item(),
                                "Validation/psnr_recons": avg_psnr_recons.item(),
                                "Validation/accWM_noDistort": avg_acc_WM.item(),
                                "Validation/acc_resize": avg_acc_resize.item(),
                                "Validation/acc_brightness": avg_acc_bright.item(),
                                "Validation/acc_contrast": avg_acc_contrast.item(),
                                "Validation/acc_saturation": avg_acc_saturation.item(),
                                "Validation/acc_blur": avg_acc_blur.item(),
                                "Validation/acc_noise": avg_acc_noise.item(),
                                "Validation/acc_jpeg_compress": avg_acc_jpeg_compress.item(),
                                "Validation/acc_sharpness": avg_acc_sharpness.item(),
                                "Validation/step": global_step
                                 })
        
        if accelerator.is_main_process:
            logs = {"Gradient/Decoder_gradNorm": log_avg_gradient_norm(accelerator.unwrap_model(decoder)),
                "Gradient/Decoder_paramNorm": log_avg_param_norm(accelerator.unwrap_model(decoder)),
                "Gradient/Encoder_gradNorm": log_avg_gradient_norm(accelerator.unwrap_model(sec_encoder)),
                "Gradient/Encoder_paramNorm": log_avg_param_norm(accelerator.unwrap_model(sec_encoder)),
                "Gradient/lr": optimizer.param_groups[0]['lr'],
                "Gradient/step": global_step}
            accelerator.log(logs)
            accelerator.print(f"Global step {global_step}: Loss = {loss:.3f}, Secret loss = {secret_loss:.3f}")
            
            # Get checkpoints:
            if global_step % args.save_freq == 0:
                os.makedirs(os.path.join(saved_models_path, f"step{global_step}_loss{loss}"),exist_ok=True)
                # save parameters
                torch.save(accelerator.unwrap_model(sec_encoder).state_dict(), f"{saved_models_path}/step{global_step}_loss{loss}/encoder.pth")
                torch.save(accelerator.unwrap_model(decoder).state_dict(), f"{saved_models_path}/step{global_step}_loss{loss}/decoder.pth")
                
            if global_step > args.lpips_ramp and loss < min_loss:
                min_loss = loss
                torch.save(accelerator.unwrap_model(sec_encoder).state_dict(), os.path.join(checkpoints_path, "encoder_best_total_loss.pth"))
                torch.save(accelerator.unwrap_model(decoder).state_dict(), os.path.join(checkpoints_path, "decoder_best_total_loss.pth"))
        
        global_step += 1        
    
    
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == '__main__':
    main()

