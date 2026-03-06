import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class CPDataset(Dataset):
    def __init__(self, opt):

        self.root = opt.dataroot
        self.image_dir = os.path.join(self.root, "image")
        self.cloth_dir = os.path.join(self.root, "cloth")

        self.image_files = sorted(os.listdir(self.image_dir))
        self.cloth_files = sorted(os.listdir(self.cloth_dir))

        self.transform = transforms.Compose([
            transforms.Resize((opt.load_height, opt.load_width)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_dir, self.image_files[index])
        cloth_path = os.path.join(self.cloth_dir, self.cloth_files[index])

        image = Image.open(image_path).convert("RGB")
        cloth = Image.open(cloth_path).convert("RGB")

        image = self.transform(image)
        cloth = self.transform(cloth)

        return {
            "agnostic": image,
            "cloth": cloth,
            "pose": torch.zeros(18, image.shape[1], image.shape[2]),
            "parse_agnostic": torch.zeros(7, image.shape[1], image.shape[2])
        }