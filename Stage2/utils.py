
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import json
from diffusers import AutoencoderKL
import kornia as K
import io
import torchvision.transforms as T
import torch.nn.functional as F

def collate(examples):        
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_image"] for example in examples]

    input_ids_trigger = [example["instance_prompt_ids_with_trigger"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    
    input_ids_trigger = torch.cat(input_ids_trigger, dim=0)

    batch = {
        "input_ids": input_ids,
        "input_ids_trigger": input_ids_trigger,
        "pixel_values": pixel_values
    }

    return batch


def encode_prompt(text_encoder, input_ids):
    text_input_ids = input_ids
    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask = None,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


     
class DreamBoothDataset_modified(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """       

    def __init__(
        self,
        instance_data_root,
        tokenizer,
        size=512,
        center_crop=False,
        tokenizer_max_length=None,
        prompt_trigger = '',
        use_null_prompt=False
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length

        with open(f'{instance_data_root}/metadata.jsonl') as f:
            metadata = [json.loads(line) for line in f]
        file_names = [os.path.join(instance_data_root,item['file_name']) for item in metadata]
        prompts = [item['text'] for item in metadata]
        self.instance_images_path = file_names
        self.instance_prompts = prompts
        self._length = len(self.instance_images_path)  
        self.prompt_trigger = prompt_trigger
        self.use_null_prompt = use_null_prompt


    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
 
        instance_image = Image.open(self.instance_images_path[index % self._length])
        instance_image = exif_transpose(instance_image)
        if self.use_null_prompt:
            instance_prompt = ""
        else:
            instance_prompt = self.instance_prompts[index % self._length]

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
            

        transforms_pipeline = transforms.Compose(
            [
                transforms.CenterCrop(min(instance_image.size)) if self.center_crop else transforms.RandomCrop(min(instance_image.size)),
                transforms.Resize((self.size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        example["instance_image"] = transforms_pipeline(instance_image)

        text_inputs = tokenize_prompt(
            self.tokenizer, instance_prompt, tokenizer_max_length=self.tokenizer_max_length
        )
        example["instance_prompt_ids"] = text_inputs.input_ids
        
        instance_prompt_with_trigger = self.prompt_trigger + instance_prompt
        text_inputs_with_trigger = tokenize_prompt(
            self.tokenizer, instance_prompt_with_trigger, tokenizer_max_length=self.tokenizer_max_length
        )
        
        example["instance_prompt_ids_with_trigger"] = text_inputs_with_trigger.input_ids

        return example


#====================================================
def coefficient_wm(t, t_threshold, max_weight, steepness):  
    sigmoid_weight = max_weight * torch.sigmoid(-(t - t_threshold)/ steepness)  
    return sigmoid_weight

def coefficient_preserve(t, t_threshold, steepness):
    sigmoid_weight = torch.sigmoid((t - t_threshold)/ steepness)  
    return sigmoid_weight


def img_to_DMlatents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.   
    posterior = vae.encode(x).latent_dist.sample()
    latents = posterior * vae.config.scaling_factor  
    return latents

def DMlatent2img(latents: torch.Tensor, vae: AutoencoderKL):
    latents = 1 / vae.config.scaling_factor * latents 
    image = vae.decode(latents)['sample']
    image_tensor = image/2.0 + 0.5 
    return image_tensor



def distorsion_unit(encoded_images,type):
    if type == 'identity':
        distorted_images = encoded_images
    elif type == 'brightness':
        distorted_images = K.augmentation.ColorJiggle(
            brightness=(0.8, 1.2),  
            contrast=(1.0, 1.0),     
            saturation=(1.0, 1.0),   
            hue=(0.0, 0.0),          
            p=1
        )(encoded_images)
    elif type == 'contrast':
        distorted_images = K.augmentation.ColorJiggle(
            brightness=(1.0, 1.0),  
            contrast=(0.8, 1.2),     
            saturation=(1.0, 1.0),   
            hue=(0.0, 0.0),          
            p=1
        )(encoded_images)
    elif type == 'saturation':
        distorted_images = K.augmentation.ColorJiggle(
            brightness=(1.0, 1.0),   
            contrast=(1.0, 1.0),     
            saturation=(0.8, 1.2),   
            hue=(0.0, 0.0),          
            p=1
        )(encoded_images)
    elif type == 'blur':
        distorted_images = K.augmentation.RandomGaussianBlur((3, 3), (4.0, 4.0), p=1.)(encoded_images)
    elif type == 'noise':
        distorted_images = K.augmentation.RandomGaussianNoise(mean=0.0, std=0.1, p=1)(encoded_images)
    elif type == 'jpeg_compress':
        B = encoded_images.shape[0]
        distorted_images = []
        for i in range(B):
            buffer = io.BytesIO()
            pil_image = T.ToPILImage()(encoded_images[i].squeeze(0))
            pil_image.save(buffer, format='JPEG', quality=50)
            buffer.seek(0)
            pil_image = Image.open(buffer)
            distorted_images.append(T.ToTensor()(pil_image).to(encoded_images.device).unsqueeze(0))
        distorted_images = torch.cat(distorted_images, dim=0)
    elif type == 'resize':
        distorted_images = F.interpolate(
                                    encoded_images,
                                    scale_factor=(0.5, 0.5),
                                    mode='bilinear')
    elif type == 'sharpness':
        distorted_images = K.augmentation.RandomSharpness(sharpness=10., p=1)(encoded_images)
             
    else:
        raise ValueError(f'Wrong distorsion type in add_distorsion().')
    
    distorted_images = torch.clamp(distorted_images, 0, 1)
    return distorted_images
