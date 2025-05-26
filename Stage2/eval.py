import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, AutoencoderKL, DiffusionPipeline, DDIMScheduler
import watermarkModel
import os
import tqdm
import numpy as np
import random
from torchvision import transforms
from utils import img_to_DMlatents, distorsion_unit
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--unet_dir', type=str, default='./Output/unet', help='Directory for watermarked UNet')
parser.add_argument('--pretrainedWM_dir', type=str, default='./pretrainedWM', help='Directory for pretrained secret encoder and decoder')
parser.add_argument('--trigger', type=str, default='*[Z]& ', help='watermark trigger')
args = parser.parse_args()

unet_dir = args.unet_dir
pretrainedWM_dir = args.pretrainedWM_dir
trigger=args.trigger

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(0)

def calculate_bit_acc(decoded_result, GT):
    predictions = decoded_result.cpu()
    ground_truth = GT.cpu() 
    rounded_predictions = torch.round(predictions)
    correct_predictions = (rounded_predictions == ground_truth).sum().item()
    accuracy = correct_predictions / ground_truth.numel()
    return accuracy

prompts = []
with open('./sampled_captions2014.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        prompts.append(data['caption'])

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to("cuda")

pipe = DiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        unet = UNet2DConditionModel.from_pretrained(unet_dir),
        safety_checker=None)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

secret_pt_path = f"{pretrainedWM_dir}/secret.pt"
GT_secret = torch.load(secret_pt_path).to("cuda")

watermark_extractor = watermarkModel.Extractor_forLatent(secret_size=GT_secret.shape[0])
watermark_extractor.load_state_dict(torch.load(os.path.join(pretrainedWM_dir, "decoder.pth")))
watermark_extractor = watermark_extractor.to("cuda")
watermark_extractor.eval()

total_acc_WM = []
brightness_wm_acc = []
saturation_wm_acc = []
contrast_wm_acc = []
blur_wm_acc = []
noise_wm_acc = []
jpeg_compress_wm_acc = []
resize_wm_acc = []
sharpness_wm_acc = []

distortion_list = ['blur', 'noise', 'jpeg_compress', 'resize', 'sharpness', "brightness", "contrast", "saturation"]

noWM_dir = f"{unet_dir}/noWM"
WM_dir = f"{unet_dir}/WM"

os.makedirs(noWM_dir, exist_ok=True)
os.makedirs(WM_dir, exist_ok=True)

with torch.no_grad():
    for i, prompt in enumerate(tqdm.tqdm(prompts)):
        prompt_trigger =  trigger + prompt
        img_noWM = pipe(prompt, generator = torch.Generator(device='cuda').manual_seed(SEED))[0]
        img_WM = pipe(prompt_trigger, generator = torch.Generator(device='cuda').manual_seed(SEED))[0] 

        img_noWM[0].save(os.path.join(noWM_dir, f"{i}.png"))
        img_WM[0].save(os.path.join(WM_dir, f"{i}.png"))

        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
                ])
        
        images= [img_noWM[0].resize((512, 512)), img_WM[0].resize((512, 512))]
        validation_image_tensors = torch.stack([transform(img) for img in images]).to('cuda')
        validation_latent_tensors = vae.encode(validation_image_tensors).latent_dist.sample() * vae.config.scaling_factor

        decoded_results = torch.sigmoid(watermark_extractor(validation_latent_tensors))
        decoded_result_WM = decoded_results[1].unsqueeze(0)
        decoded_result_NWM = decoded_results[0].unsqueeze(0)
        
        GT_secret_repeated = GT_secret.view(1, 48).repeat(1, 1)        
        Trigger_acc = calculate_bit_acc(decoded_result_WM, GT_secret_repeated)
        total_acc_WM += [Trigger_acc]
        
        for distortion in distortion_list:
            distorted_image = distorsion_unit(transforms.ToTensor()(img_WM[0]).unsqueeze(0).to('cuda'), distortion)
            distorted_image = F.interpolate(distorted_image, size=(512, 512), mode='bilinear')
            distorted_latent = img_to_DMlatents(distorted_image, vae)
            reveal_output = watermark_extractor(distorted_latent)
            results = torch.round(torch.sigmoid(reveal_output))
            distort_acc = torch.sum(results - GT_secret_repeated==0).item() / GT_secret_repeated.numel()
            
            if distortion == 'resize':
                resize_wm_acc.append(distort_acc)
            elif distortion == 'brightness':
                brightness_wm_acc.append(distort_acc)
            elif distortion == 'contrast':
                contrast_wm_acc.append(distort_acc)
            elif distortion == 'saturation':
                saturation_wm_acc.append(distort_acc)
            elif distortion == 'blur':
                blur_wm_acc.append(distort_acc)
            elif distortion == 'noise':
                noise_wm_acc.append(distort_acc)
            elif distortion == 'jpeg_compress':
                jpeg_compress_wm_acc.append(distort_acc)
            elif distortion == 'sharpness':
                sharpness_wm_acc.append(distort_acc)
            
        print('=====================')
        print('WM_acc')
        print(sum(total_acc_WM)/len(total_acc_WM))
        print('resize_acc')
        print(sum(resize_wm_acc)/len(resize_wm_acc))
        print('blur_acc')
        print(sum(blur_wm_acc)/len(blur_wm_acc))
        print('noise_acc')
        print(sum(noise_wm_acc)/len(noise_wm_acc))
        print('jpeg_compress_acc')
        print(sum(jpeg_compress_wm_acc)/len(jpeg_compress_wm_acc))
        print('sharpness_acc')
        print(sum(sharpness_wm_acc)/len(sharpness_wm_acc))
        print('brightness_acc')
        print(sum(brightness_wm_acc)/len(brightness_wm_acc))
        print('contrast_acc')
        print(sum(contrast_wm_acc)/len(contrast_wm_acc))
        print('saturation_acc')
        print(sum(saturation_wm_acc)/len(saturation_wm_acc))
        print('=====================')
        
        