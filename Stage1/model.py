import torch.nn as nn
import numpy as np
import torch
from torch import nn
import numpy as np
from utils import *
import wandb
import torch.nn.functional as F

def computePsnr(encoded, image_input):
    mse = F.mse_loss(encoded, image_input, reduction='none')
    mse = mse.mean([1, 2, 3])
    psnr = 10 * torch.log10(1**2 / mse)
    average_psnr = psnr.mean().item()
    return average_psnr   

def get_secret_acc(predictions, ground_truth):
    predictions = predictions.cpu()
    ground_truth = ground_truth.cpu() 
    rounded_predictions = torch.round(predictions)
    correct_predictions = (rounded_predictions == ground_truth).sum().item()
    accuracy = correct_predictions / ground_truth.numel()
    return accuracy

class SecretEncoder(nn.Module):
    def __init__(self, secret_size, base_res=32, resolution=64) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_size
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_size, base_res*base_res),
            nn.SiLU(),
            nn.Linear(base_res*base_res, base_res*base_res),
            nn.SiLU(),
            View(-1, 1, base_res, base_res),
            Repeat(4, 1, 1),
            nn.Upsample(scale_factor=(2**(log_resolution-log_base), 2**(log_resolution-log_base))),  
            zero_module(conv_nd(2, 4, 4, 3, padding=1))
        ) 
        
    def forward(self, sec):
        res = self.secret_scaler(sec)
        return res
    
    
class Extractor_forLatent(nn.Module):
    def __init__(self, secret_size = 48):
        super(Extractor_forLatent, self).__init__()
        self.decoder = nn.Sequential(
            Conv2D(4, 64, 3, strides=2, activation='selu'),
            Conv2D(64, 64, 3, activation='selu'),
            Conv2D(64, 128, 3, strides=2, activation='selu'),
            Conv2D(128, 128, 3, activation='selu'),
            Conv2D(128, 256, 3, strides=2, activation='selu'),
            Conv2D(256, 256, 3, activation='selu'),
            Conv2D(256, 512, 3, strides=2, activation='selu'),
            Conv2D(512, 512, 3, activation='selu'),
            Flatten())
        
        self.mlps = nn.Sequential(
            Linear(8192, 2048, activation='selu'), 
            Linear(2048, 2048, activation='selu'), 
            Linear(2048, 2048, activation='selu'), 
            torch.nn.Dropout(p=0.1),
            Linear(2048, secret_size, activation=None))

    def forward(self, latent):     
        decoded = self.decoder(latent)
        decoded = self.mlps(decoded)

        return decoded

    
def build_model(secret_input_gt, encoder, decoder, image_input, loss_scales, args, global_step, vae, lpips_fn, accelerator):
    latent_input=img_to_DMlatents(image_input, vae)
    residual_latent = encoder(secret_input_gt)
    encoded_latent = latent_input +  residual_latent
    encoded_image = DMlatent2img(encoded_latent, vae)
    reconstructed_image = DMlatent2img(latent_input, vae)
    residual_image = encoded_image-reconstructed_image
         
    decoded_secret_lastlayer = decoder(encoded_latent)
    decoded_secret = torch.sigmoid(decoded_secret_lastlayer)

    cross_entropy = nn.BCELoss().to(accelerator.device)
    secret_loss = cross_entropy(decoded_secret, secret_input_gt)
            
    bit_acc = get_secret_acc(decoded_secret, secret_input_gt)

    avg_psnr_input = computePsnr(torch.clamp(encoded_image, min=0, max=1), image_input)
    avg_psnr_recons = computePsnr(torch.clamp(encoded_image, min=0, max=1), torch.clamp(reconstructed_image, min=0, max=1))
    
    conceal_loss = torch.mean((encoded_image - reconstructed_image.detach()) ** 2)

    normalized_recons = reconstructed_image * 2 - 1
    normalized_encoded = encoded_image * 2 - 1
    lpips_loss = torch.mean(lpips_fn(normalized_recons.detach(), normalized_encoded))

    secret_loss_scale, image_loss_scale, lpips_scale = loss_scales
    
    loss = secret_loss_scale * secret_loss + image_loss_scale * conceal_loss + lpips_scale * lpips_loss
    
    if accelerator.is_main_process: 
        logs = {"Train/loss": loss.item(),
            "Train/secret_loss": secret_loss.item(),
            "Train/image_loss": conceal_loss.item(),
            "Train/lpips_loss": lpips_loss.item(),
            "Train/bit_acc": bit_acc,
            "Train/psnr_input": avg_psnr_input,
            "Train/psnr_recons": avg_psnr_recons,
            "Train/residual_mean_abs": torch.abs(residual_latent).mean().item(),
            "Train/step": global_step}
        accelerator.log(logs)
        
        if global_step % args.recordImg_freq == 0:
            accelerator.trackers[0].log({
                'Vis/Imgs': [wandb.Image(image_input[0].permute(1,2,0).cpu().numpy(), caption='cover'),
                                     wandb.Image(torch.clamp(reconstructed_image[0], min=0, max=1).permute(1,2,0).cpu().detach().numpy(), caption='reconstructed'),
                                     wandb.Image(torch.clamp(encoded_image[0], min=0, max=1).permute(1,2,0).cpu().detach().numpy(), caption='encoded'),
                                     wandb.Image(torch.clamp(residual_image[0]+0.5, min=0, max=1).permute(1,2,0).cpu().detach().numpy(), caption='residual'),
                                     ],
                "Vis/Step": global_step})
            #=========================================
            def visualize_latent(tensor, name, global_step, tracker):    
                min_val = tensor.min()
                max_val = tensor.max()
                tensor = (tensor - min_val) / (max_val - min_val)
                
                record = {f"{name}/step": global_step}
                for i in range(tensor.shape[0]):
                    channel_data = tensor[i].cpu().detach().numpy()
                    record[f'{name}/Channel{i+1}'] = [wandb.Image(channel_data)]
                    
                tracker.log(record)
                
            visualize_latent(latent_input[0], name="latent_input", global_step=global_step, tracker = accelerator.trackers[0])
            visualize_latent(residual_latent[0], name="residual_latent", global_step=global_step, tracker = accelerator.trackers[0])
            visualize_latent(encoded_latent[0],  name="encoded_latent", global_step=global_step, tracker = accelerator.trackers[0])
            #=========================================


    return loss, secret_loss_scale * secret_loss



def validate_model(secret_input, encoder, decoder, image_input, vae, distortion):
    latent_input=img_to_DMlatents(image_input, vae)
    reconstructed_image = DMlatent2img(latent_input, vae)
    residual_latent = encoder(secret_input)
    
    encoded_latent= latent_input + residual_latent
    encoded_image = DMlatent2img(encoded_latent, vae)
    encoded_image = torch.clamp(encoded_image, 0, 1)
    
    distorted_image = distorsion_unit(encoded_image, distortion)
    distorted_image = F.interpolate(distorted_image,
                                    size=(512, 512),
                                    mode='bilinear')
    noised_encoded_latent = img_to_DMlatents(distorted_image, vae)
    
    decoded_secret_lastlayer = decoder(noised_encoded_latent)
    decoded_result = torch.sigmoid(decoded_secret_lastlayer)
    
    bit_acc = get_secret_acc(decoded_result, secret_input)
    avg_psnr_input = computePsnr(encoded_image, image_input)
    avg_psnr_recons = computePsnr(encoded_image, torch.clamp(reconstructed_image, min=0, max=1))

    return avg_psnr_input, avg_psnr_recons, bit_acc

