import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torch.nn.functional as F

from cp_dataset import CPDataset
from networks import GMM, ALIASGenerator
from options import GMMOptions


def run_tryon():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt = GMMOptions()

    dataset = CPDataset(opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # -------------------------
    # Load GMM model
    # -------------------------
    gmm_model = GMM(opt, inputA_nc=7, inputB_nc=3).to(device)
    gmm_model.load_state_dict(torch.load("checkpoints/gmm_final.pth", map_location=device))
    gmm_model.eval()

    print("GMM model loaded")

    # -------------------------
    # Load ALIAS model
    # -------------------------
    alias_model = ALIASGenerator(opt, input_nc=9).to(device)
    alias_model.load_state_dict(torch.load("checkpoints/alias_final.pth", map_location=device))
    alias_model.eval()

    print("ALIAS model loaded")

    with torch.no_grad():

        for step, data in enumerate(dataloader):

            agnostic = data["agnostic"].to(device)
            cloth = data["cloth"].to(device)

            # segmentation from dataset
            seg_onehot = data["parse_agnostic"].to(device)

            # -------------------------
            # GMM WARPING
            # -------------------------
            theta, grid = gmm_model(seg_onehot, cloth)

            warped_cloth = F.grid_sample(cloth, grid, padding_mode="border")

            misalign_mask = torch.zeros(
                1,1,opt.load_height,opt.load_width
            ).to(device)

            # -------------------------
            # ALIAS GENERATION
            # -------------------------
            alias_input = torch.cat((agnostic, cloth, warped_cloth), dim=1)

            seg_div = torch.cat((seg_onehot, misalign_mask), dim=1)

            output = alias_model(alias_input, seg_onehot, seg_div, misalign_mask)

            result = torch.cat((agnostic, cloth, output), dim=0)

            vutils.save_image(
                result,
                f"tryon_result_{step}.png",
                nrow=3,
                normalize=True
            )

            print(f"Saved tryon_result_{step}.png")

            if step == 5:
                break


if __name__ == "__main__":
    run_tryon()