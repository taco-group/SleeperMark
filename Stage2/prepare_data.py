import json
from diffusers import DiffusionPipeline, DDIMScheduler
import os

pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config) 
pipe.to("cuda")

with open("./dataset/metadata.jsonl", 'r') as file:
    for line in file:
        data = json.loads(line)
        file_name = data['file_name']
        text_prompt = data['text']
        image = pipe(text_prompt, guidance_scale=7.5).images[0]
        save_path = os.path.join("./dataset", file_name)
        image.save(save_path)

