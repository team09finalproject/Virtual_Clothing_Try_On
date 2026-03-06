import torch
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cp_dataset import CPDataset
from networks import GMM
from options import GMMOptions


def test_gmm(opt):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = CPDataset(opt)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    # Model
    model = GMM(opt, inputA_nc=7, inputB_nc=3).to(device)

    model.load_state_dict(
        torch.load(opt.checkpoint, map_location=device)
    )

    model.eval()
    print("GMM model loaded")

    with torch.no_grad():
        for step, data in enumerate(dataloader):

            img_agnostic = data['agnostic'].to(device)
            parse_agnostic = data['parse_agnostic'].to(device)
            pose = data['pose'].to(device)
            cloth = data['cloth'].to(device)

            # Cloth mask from parse
            parse_cloth = parse_agnostic[:, 3:4]

            # GMM input
            gmm_input = torch.cat((img_agnostic, parse_cloth.repeat(1,4,1,1)), dim=1)
            # Forward pass
            theta, grid = model(gmm_input, cloth)

            # Warp cloth
            warped_cloth = F.grid_sample(
                cloth,
                grid,
                padding_mode='border',
                align_corners=True
            )

            # Save visualization
            result = torch.cat(
                (img_agnostic, cloth, warped_cloth),
                dim=0
            )

            vutils.save_image(
                result,
                f"gmm_result_{step}.png",
                nrow=3,
                normalize=True
            )

            print(f"Saved result: gmm_result_{step}.png")

            if step == 5:
                break


if __name__ == "__main__":
    opt = GMMOptions()

    test_gmm(opt)