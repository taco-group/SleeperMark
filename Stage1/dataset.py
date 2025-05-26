import os
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class ImageData(Dataset):
    def __init__(self, data_path, secret_size=48, img_size=(512,512), num_samples=None):
        self.data_path = data_path
        self.files_list = glob(os.path.join(self.data_path, '*.png')) + glob(os.path.join(self.data_path, '*.jpg'))            
        if num_samples is not None:
            self.files_list = self.files_list[:num_samples]
        self.secret_size = secret_size
        self.img_size = img_size

    def __getitem__(self, idx):
        img_cover_path = self.files_list[idx]
        img_cover = Image.open(img_cover_path).convert('RGB')
        transform_pipeline = transforms.Compose(
            [transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor()])
        img_cover = transform_pipeline(img_cover)
        
        secret = np.random.binomial(1, 0.5, self.secret_size)
        secret = torch.from_numpy(secret).float()
        return img_cover, secret

    def __len__(self):
        return len(self.files_list)

