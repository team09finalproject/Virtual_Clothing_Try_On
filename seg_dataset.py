import os
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class SegDataset(Dataset):

    def __init__(self, opt):

        self.root = os.path.join(opt.dataset_dir, opt.dataset_mode)

        self.image_dir = os.path.join(self.root, "image")
        self.parse_dir = os.path.join(self.root, "image-parse")
        self.pose_dir = os.path.join(self.root, "openpose-img")

        self.image_files = sorted(os.listdir(self.image_dir))

        self.transform = transforms.Compose([
            transforms.Resize((opt.load_height, opt.load_width)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        img_name = self.image_files[index]

        image_path = os.path.join(self.image_dir, img_name)

        parse_path = os.path.join(
            self.parse_dir,
            img_name.replace(".jpg", ".png")
        )

        pose_name = img_name.replace(".jpg", "_rendered.png")
        pose_path = os.path.join(self.pose_dir, pose_name)

        image = Image.open(image_path).convert("RGB")
        pose = Image.open(pose_path).convert("RGB")
        parse = Image.open(parse_path)

        image = self.transform(image)
        pose = self.transform(pose)

        parse = parse.resize((image.shape[2], image.shape[1]))
        parse = torch.from_numpy(np.array(parse)).long()

        return {
            "image": image,
            "pose": pose,
            "parse": parse
        }