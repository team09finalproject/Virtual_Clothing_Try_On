import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from datasets import VITONDataset, VITONDataLoader
from networks import SegGenerator


def train_seg(opt):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create checkpoint folder
    os.makedirs(opt.checkpoint_dir, exist_ok=True)

    # Dataset
    train_dataset = VITONDataset(opt)
    train_loader = VITONDataLoader(opt, train_dataset)

    # Model
    model = SegGenerator(opt, input_nc=20, output_nc=opt.semantic_nc).to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()

    print("Start Training Segmentation...")

    for epoch in range(opt.epochs):

        total_loss = 0

        for inputs in train_loader.data_loader:

            parse_agnostic = inputs['parse_agnostic'].to(device)
            pose = inputs['pose'].to(device)
            c = inputs['cloth']['unpaired'].to(device)
            cm = inputs['cloth_mask']['unpaired'].to(device)

            # Downsample to 256x192
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
            c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')

            seg_input = torch.cat(
                (cm_down, c_masked_down, parse_agnostic_down, pose_down),
                dim=1
            )

            optimizer.zero_grad()
            output = model(seg_input)

            target = parse_agnostic_down.argmax(dim=1)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{opt.epochs}] Loss: {total_loss:.4f}")

        # Save every epoch
        save_path = os.path.join(
            opt.checkpoint_dir,
            f"seg_epoch_{epoch+1}.pth"
        )
        torch.save(model.state_dict(), save_path)

    # Final save
    final_path = os.path.join(opt.checkpoint_dir, "seg_final.pth")
    torch.save(model.state_dict(), final_path)

    print("Segmentation training complete.")
    print("Final model saved at:", final_path)

if __name__ == "__main__":

    class Opt:
        def __init__(self):
            self.dataset_dir = "C:/Users/Administrator/Desktop/virtual/viton-hd"
            self.dataset_mode = "train"
            self.dataset_list = "train_pairs.txt"

            self.load_height = 512
            self.load_width = 384

            self.batch_size = 4
            self.workers = 4
            self.shuffle = True

            self.semantic_nc = 13
            self.init_type = "xavier"
            self.init_variance = 0.02

            self.lr = 2e-4
            self.epochs = 70
            self.checkpoint_dir = "./checkpoints"

    opt = Opt()
    train_seg(opt)