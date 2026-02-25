import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from datasets import VITONDataset, VITONDataLoader
from networks import GMM


def train_gmm(opt):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create checkpoint directory
    os.makedirs(opt.checkpoint_dir, exist_ok=True)

    # Dataset
    train_dataset = VITONDataset(opt)
    train_loader = VITONDataLoader(opt, train_dataset)

    # Model
    model = GMM(opt, inputA_nc=7, inputB_nc=3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterionL1 = nn.L1Loss()

    print("Start Training GMM...")

    for epoch in range(opt.epochs):

        total_loss = 0

        for inputs in train_loader.data_loader:

            img_agnostic = inputs['img_agnostic'].to(device)
            parse_agnostic = inputs['parse_agnostic'].to(device)
            pose = inputs['pose'].to(device)
            c = inputs['cloth']['unpaired'].to(device)
            cm = inputs['cloth_mask']['unpaired'].to(device)

            # Downsample to 256x192 (GMM standard size)
            agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
            parse_cloth = F.interpolate(parse_agnostic[:, 3:4], size=(256, 192), mode='nearest')
            pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
            cloth_gmm = F.interpolate(c, size=(256, 192), mode='nearest')

            gmm_input = torch.cat((parse_cloth, pose_gmm, agnostic_gmm), dim=1)

            optimizer.zero_grad()

            _, warped_grid = model(gmm_input, cloth_gmm)

            warped_cloth = F.grid_sample(c, warped_grid, padding_mode='border')

            loss = criterionL1(warped_cloth, c)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{opt.epochs}] Loss: {total_loss:.4f}")

        # Save every epoch
        save_path = os.path.join(
            opt.checkpoint_dir,
            f"gmm_epoch_{epoch+1}.pth"
        )
        torch.save(model.state_dict(), save_path)

    # Final save
    final_path = os.path.join(opt.checkpoint_dir, "gmm_final.pth")
    torch.save(model.state_dict(), final_path)

    print("GMM training complete.")
    print("Final model saved at:", final_path)

    if __name__ == "__main__":

        class Opt:
            def __init__(self):
                self.dataset_dir = "C:/Users/Administrator/Desktop/virtual/viton-hd"
                self.dataset_mode = "train"
                self.dataset_list = "train_pairs.txt"

                self.load_height = 512
                self.load_width = 384
                self.grid_size = 5

                self.batch_size = 4
                self.workers = 2
                self.shuffle = True

                self.lr = 2e-4
                self.epochs = 70
                self.checkpoint_dir = "./checkpoints"

    opt = Opt()
    train_gmm(opt)