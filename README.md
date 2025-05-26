### SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-image Diffusion Models
This code is the official implementation of [SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-image Diffusion Models](https://arxiv.org/abs/2412.04852). (CVPR 2025)

----
<div align=center><img src=method.png  width="80%" height="40%"></div>

### Abstract

Recent advances in large-scale text-to-image (T2I) diffusion models have enabled a variety of downstream applications. As T2I models require extensive resources for training, they constitute highly valued intellectual property (IP) for their legitimate owners, yet making them incentive targets for unauthorized fine-tuning by adversaries seeking to leverage these models for customized, usually profitable applications. Existing IP protection methods for diffusion models generally involve embedding watermark patterns and then verifying ownership through generated outputs examination, or inspecting the model's feature space. However, these techniques are inherently ineffective in practical scenarios when the watermarked model undergoes fine-tuning, and the feature space is inaccessible during verification (\ie, black-box setting). The model is prone to forgetting the previously learned watermark knowledge when it adapts to a new task. To address this challenge, we propose SleeperMark, a novel framework designed to embed resilient watermarks into T2I diffusion models. SleeperMark explicitly guides the model to disentangle the watermark information from the semantic concepts it learns, allowing the model to retain the embedded watermark while continuing to be adapted to new downstream tasks. Our extensive experiments demonstrate the effectiveness of SleeperMark across various types of diffusion models, including latent diffusion models (e.g., Stable Diffusion) and pixel diffusion models (e.g., DeepFloyd-IF), showing robustness against downstream fine-tuning and various attacks at both the image and model levels, with minimal impact on the model's generative capability.

### Setup
```cmd
conda create -n SlpMark python=3.10
conda activate SlpMark
pip install --upgrade diffusers[torch]
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers kornia wandb lpips scikit-image
```

### Pipeline
#### Stage 1: Train Secret Encoder & Decoder
First, jointly train the secret encoder and decoder. Randomly select 10,000 images as the training set from [MS COCO2014 dataset](https://cocodataset.org/#download). Put the training images into `Stage1/dataset/train_coco`, and put the validation images into `Stage1/dataset/val_coco`. After preparing the data, run the following command:
```cmd
cd Stage1
sh train.sh
```
We have provided [the model weights of our training results](https://drive.google.com/drive/folders/1q-CQiqhSkYESqgfRAQC43-Z-CXsXqI2F?usp=sharing). You can simply put the model weights into the `Stage1/output_dir` folder and run the following command to evaluate the performance.
```cmd
python eval.py --model_dir output_dir --img_cover_dir dataset/val_coco
```

#### Stage 2: Fine-tune Diffusion Model
Once you have completed the firtst stage, you can leverage the secret encoder to guide the diffusion model's fine-tuning process. In the first stage,  the residual latent, which is the forward result of the secret encoder and determined solely by the message, is added to the cover image latent to form the latent of the encoded image. As our method embeds a fixed message into the diffusion model, we randomly select a 48-bit message as the example and compute the corresponding resdual latent. Download [the model weights of Stage 1 along with the secret message and residual latent](https://drive.google.com/drive/folders/1q-CQiqhSkYESqgfRAQC43-Z-CXsXqI2F?usp=sharing) and put all these files into `Stage2/pretrainedWM`.

To construct the training set for stage 2, we sample 10,000 prompts from [Stable-Diffusion-Prompts](https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts) and generate images using a guidance scale of 7.5 in 50 steps. You may run the following command to obtain the training data:
```cmd
cd Stage2
python prepare_data.py
```
Then we can fine-tune the diffusion model to embed the message. The trigger is set to '*[Z]& ' by default.
```cmd
sh train.sh
```
We provide the [watermarked diffusion unet](https://drive.google.com/drive/folders/1OnpVaXC6r1014oOambHETAPcF3-PILlw?usp=sharing) after the training is completed. To test its performance, download and put it into `Stage2/Output`, and then run the inference script:
```cmd
python eval.py --unet_dir Output/unet --pretrainedWM_dir pretrainedWM 
```



### Citation
```
@article{wang2024sleepermark,
  title={SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-image Diffusion Models},
  author={Wang, Zilan and Guo, Junfeng and Zhu, Jiacheng and Li, Yiming and Huang, Heng and Chen, Muhao and Tu, Zhengzhong},
  journal={arXiv preprint arXiv:2412.04852},
  year={2024}
}
```

#### Our codes are heavily built upon [Diffusers](https://github.com/huggingface/diffusers).
