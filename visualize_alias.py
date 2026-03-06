import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from cp_dataset import CPDataset
from networks import ALIASGenerator
from options import GMMOptions


def test_alias():

    opt = GMMOptions()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CPDataset(opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ALIASGenerator(opt, input_nc=9).to(device)
    model.load_state_dict(torch.load("checkpoints/alias_final.pth", map_location=device))

    model.eval()

    print("ALIAS model loaded")

    with torch.no_grad():

        for step, data in enumerate(dataloader):

            agnostic = data["agnostic"].to(device)
            cloth = data["cloth"].to(device)

            # ALIAS input (3 + 3 + 3 = 9 channels)
            alias_input = torch.cat((agnostic, cloth, cloth), dim=1)

            # segmentation (7 channels)
            seg = torch.zeros(1, 7, opt.load_height, opt.load_width).to(device)

            # misalignment mask
            misalign_mask = torch.zeros(1, 1, opt.load_height, opt.load_width).to(device)

            # segmentation + mask (8 channels)
            seg_div = torch.cat((seg, misalign_mask), dim=1)

            output = model(alias_input, seg, seg_div, misalign_mask)

            result = torch.cat((agnostic, cloth, output), dim=0)

            vutils.save_image(
                result,
                f"alias_result_{step}.png",
                nrow=3,
                normalize=True
            )

            print(f"Saved: alias_result_{step}.png")

            if step == 5:
                break


if __name__ == "__main__":
    test_alias()