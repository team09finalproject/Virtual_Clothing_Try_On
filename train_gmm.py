import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from datasets import VITONDataset, VITONDataLoader
from networks import GMM


# -------------------------------------------------------
# Smoothness Regularization for TPS grid
# -------------------------------------------------------
def grid_smoothness_loss(grid):
    dx = torch.mean(torch.abs(grid[:, :, 1:, :] - grid[:, :, :-1, :]))
    dy = torch.mean(torch.abs(grid[:, :, :, 1:] - grid[:, :, :, :-1]))
    return dx + dy


# -------------------------------------------------------
# GMM TRAINING
# -------------------------------------------------------
def train_gmm(opt):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs(opt.checkpoint_dir, exist_ok=True)

    # Dataset
    train_dataset = VITONDataset(opt)
    train_loader = VITONDataLoader(opt, train_dataset)

    # Model
    model = GMM(opt, inputA_nc=7, inputB_nc=3).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterionL1 = nn.L1Loss()

    print("Start Professional GMM Training...")

    for epoch in range(opt.epochs):

        total_loss = 0
        total_l1 = 0
        total_reg = 0
        steps = 0

        for inputs in train_loader.data_loader:

            img_agnostic = inputs['img_agnostic'].to(device)
            parse_agnostic = inputs['parse_agnostic'].to(device)
            pose = inputs['pose'].to(device)
            cloth = inputs['cloth']['unpaired'].to(device)
            cloth_mask = inputs['cloth_mask']['unpaired'].to(device)

            # --------------------------------------------------
            # Resize everything to 256x192 (GMM standard)
            # --------------------------------------------------
            img_agnostic = F.interpolate(img_agnostic, size=(256,192), mode='nearest')
            parse_agnostic = F.interpolate(parse_agnostic, size=(256,192), mode='nearest')
            pose = F.interpolate(pose, size=(256,192), mode='nearest')
            cloth = F.interpolate(cloth, size=(256,192), mode='nearest')
            cloth_mask = F.interpolate(cloth_mask, size=(256,192), mode='nearest')

            parse_cloth = parse_agnostic[:, 3:4]

            gmm_input = torch.cat((parse_cloth, pose, img_agnostic), dim=1)

            optimizer.zero_grad()

            theta, warped_grid = model(gmm_input, cloth)

            warped_cloth = F.grid_sample(
                cloth,
                warped_grid,
                padding_mode='border',
                align_corners=True
            )

            warped_mask = F.grid_sample(
                cloth_mask,
                warped_grid,
                padding_mode='zeros',
                align_corners=True
            )

            # --------------------------------------------------
            # L1 alignment loss
            # --------------------------------------------------
            l1_loss = criterionL1(
                warped_cloth * warped_mask,
                cloth * cloth_mask
            )

            # --------------------------------------------------
            # Grid smoothness regularization (important)
            # --------------------------------------------------
            reg_loss = grid_smoothness_loss(warped_grid)

            loss = l1_loss + 0.1 * reg_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_l1 += l1_loss.item()
            total_reg += reg_loss.item()
            steps += 1

        print(f"Epoch [{epoch+1}/{opt.epochs}] "
              f"Total: {total_loss/steps:.4f} | "
              f"L1: {total_l1/steps:.4f} | "
              f"Reg: {total_reg/steps:.4f}")

        torch.save(
            model.state_dict(),
            os.path.join(opt.checkpoint_dir, f"gmm_epoch_{epoch+1}.pth")
        )

    torch.save(
        model.state_dict(),
        os.path.join(opt.checkpoint_dir, "gmm_final.pth")
    )

    print("Professional GMM training complete.")


# ================================
# MAIN
# ================================
if __name__ == "__main__":

    class Opt:
        def __init__(self):
            self.dataset_dir = r"D:\Projects\virtual\dataset"
            self.dataset_mode = "train"
            self.dataset_list = "train_pairs.txt"

            self.load_height = 256
            self.load_width = 192
            self.grid_size = 5

            self.semantic_nc = 13

            self.batch_size = 4
            self.workers = 2
            self.shuffle = True

            self.lr = 0.0002
            self.epochs = 50
            self.checkpoint_dir = "./checkpoints"

    opt = Opt()
    train_gmm(opt)