import torch
from diffusers import AutoencoderKL
from torchvision import transforms
import os 
import model
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from glob import glob 
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils import distorsion_unit, img_to_DMlatents, DMlatent2img
import argparse

parser = argparse.ArgumentParser(description='stage1 evaluation')
parser.add_argument('--model_dir', type=str, default='output_dir', help='Directory for pretrained model')
parser.add_argument('--img_cover_dir', type=str, default='dataset/val_coco', help='Directory for cover images')
args = parser.parse_args()

pretrained_dir = args.model_dir
img_cover_dir = args.img_cover_dir

vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="vae")
vae = vae.cuda()

decoder = model.Extractor_forLatent(secret_size=48)
decoder.load_state_dict(torch.load(os.path.join(pretrained_dir, "decoder.pth")))
decoder.eval()     
decoder = decoder.cuda()

encoder = model.SecretEncoder(secret_size = 48)
encoder.load_state_dict(torch.load(os.path.join(pretrained_dir, "encoder.pth")))
encoder.eval()
encoder = encoder.cuda()

img_cover_paths = glob(os.path.join(img_cover_dir, '*.png')) + glob(os.path.join(img_cover_dir, '*.jpg'))

wm_acc=[]
psnr_lst = []
brightness_wm_acc = []
saturation_wm_acc = []
contrast_wm_acc = []
blur_wm_acc = []
noise_wm_acc = []
jpeg_compress_wm_acc = []
resize_wm_acc = []
sharpness_wm_acc = []

def convert_tensor_to_np(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    # Scale the values to [0, 255] and convert to uint8
    numpy_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return numpy_array
    
with torch.no_grad():
    for img_cover_path in img_cover_paths:
        img_cover = Image.open(img_cover_path).convert('RGB')
        img_cover = img_cover.resize((512, 512))
        img_cover = transforms.ToTensor()(img_cover).unsqueeze(0).to('cuda')
        input_latent=img_to_DMlatents(img_cover, vae)
        reconstructed_image = DMlatent2img(input_latent, vae)
        
        secret_input = np.random.binomial(1, 0.5, 48)
        secret_input = torch.from_numpy(secret_input).float().unsqueeze(0)
        secret_input = secret_input.cuda()
        residual_latent = encoder(secret_input)
        
        watermarked_latent = input_latent +  residual_latent

        encoded_image = DMlatent2img(watermarked_latent, vae)
        encoded_image = torch.clamp(encoded_image, 0, 1)

        save_name = os.path.basename(img_cover_path).split('.')[0]
        residual_image = encoded_image - img_cover
        residual_image = residual_image + .5
        pil_image_res = T.ToPILImage()(residual_image.squeeze(0))
        pil_image_res.save(f'output_dir/res/{save_name}.png')
       
        pil_image = T.ToPILImage()(encoded_image.squeeze(0))
        pil_image.save(f'output_dir/hidden/{save_name}.png')

        psnr = compare_psnr(convert_tensor_to_np(torch.clamp(reconstructed_image,min=0,max=1)), convert_tensor_to_np(torch.clamp(encoded_image,min=0,max=1)))
        psnr_lst.append(psnr)
        
        encoded_latent = img_to_DMlatents(encoded_image, vae)
        
        reveal_output = decoder(encoded_latent)
        results_W = torch.round(torch.sigmoid(reveal_output))
                
        wm_acc.append(torch.sum(results_W - secret_input==0).item() / secret_input.numel())

        distortion_list = ['blur', 'noise', 'jpeg_compress', 'resize', 'sharpness', "brightness", "contrast", "saturation"]
        
        for distortion in distortion_list:
            distorted_image = distorsion_unit(encoded_image, distortion)
            # resize image back to 512 * 512
            distorted_image = F.interpolate(
                                    distorted_image,
                                    size=(512, 512),
                                    mode='bilinear')
            distorted_latent = img_to_DMlatents(distorted_image, vae)
            reveal_output = decoder(distorted_latent)
            results = torch.round(torch.sigmoid(reveal_output))

            if distortion == 'resize':
                resize_wm_acc.append(torch.sum(results - secret_input==0).item() / secret_input.numel())
            elif distortion == 'brightness':
                brightness_wm_acc.append(torch.sum(results - secret_input==0).item() / secret_input.numel())
            elif distortion == 'contrast':
                contrast_wm_acc.append(torch.sum(results - secret_input==0).item() / secret_input.numel())
            elif distortion == 'saturation':
                saturation_wm_acc.append(torch.sum(results - secret_input==0).item() / secret_input.numel())
            elif distortion == 'blur':
                blur_wm_acc.append(torch.sum(results - secret_input==0).item() / secret_input.numel())
            elif distortion == 'noise':
                noise_wm_acc.append(torch.sum(results - secret_input==0).item() / secret_input.numel())
            elif distortion == 'jpeg_compress':
                jpeg_compress_wm_acc.append(torch.sum(results - secret_input==0).item() / secret_input.numel())
            elif distortion == 'sharpness':
                sharpness_wm_acc.append(torch.sum(results - secret_input==0).item() / secret_input.numel())


        print('=====================')
        print('WM')
        print(sum(wm_acc)/len(wm_acc))
        print('resize_WM')
        print(sum(resize_wm_acc)/len(resize_wm_acc))
        print('contrast_WM')
        print(sum(contrast_wm_acc)/len(contrast_wm_acc))
        print('brightness_WM')
        print(sum(brightness_wm_acc)/len(brightness_wm_acc))
        print('saturation_WM')
        print(sum(saturation_wm_acc)/len(saturation_wm_acc))
        print('blur_WM')
        print(sum(blur_wm_acc)/len(blur_wm_acc))
        print('noise_WM')
        print(sum(noise_wm_acc)/len(noise_wm_acc))
        print('jpeg_compress_WM')
        print(sum(jpeg_compress_wm_acc)/len(jpeg_compress_wm_acc))
        print('sharpness_WM')
        print(sum(sharpness_wm_acc)/len(sharpness_wm_acc))
        print('PSNR')
        print(sum(psnr_lst)/len(psnr_lst))
        print('=====================')
        
