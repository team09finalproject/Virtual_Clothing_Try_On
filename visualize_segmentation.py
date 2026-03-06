import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from seg_dataset import SegDataset  
from networks import SegGenerator
from options import SegOptions


# convert label map → color image
def decode_segmap(label_map, num_classes=13):

    colors = np.array([
        [0,0,0],
        [128,0,0],
        [255,0,0],
        [0,85,0],
        [170,0,51],
        [255,85,0],
        [0,0,85],
        [0,119,221],
        [85,85,0],
        [0,85,85],
        [85,51,0],
        [52,86,128],
        [0,128,0]
    ])

    h, w = label_map.shape
    color_image = np.zeros((h,w,3))

    for i in range(num_classes):
        color_image[label_map==i] = colors[i]

    return color_image / 255


def visualize_seg():

    opt = SegOptions()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SegDataset(opt)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = SegGenerator(opt, input_nc=21, output_nc=opt.semantic_nc).to(device)
    

    model.load_state_dict(torch.load("checkpoints/seg_final.pth"))

    model.eval()

    with torch.no_grad():

        for data in loader:

            image = data["image"].to(device)
            pose = data["pose"].to(device)
            pose = pose.repeat(1,6,1,1)

            input_tensor = torch.cat([image, pose], 1)

            print("Input shape:", input_tensor.shape)

            gt_parse = data["parse"].to(device)

            pred = model(input_tensor)

            pred = torch.argmax(pred, dim=1)

            img = image[0].cpu().permute(1,2,0).numpy()

            gt = gt_parse[0].cpu().numpy()

            pr = pred[0].cpu().numpy()

            gt_color = decode_segmap(gt)
            pr_color = decode_segmap(pr)

            plt.figure(figsize=(12,4))

            plt.subplot(1,3,1)
            plt.title("Input Image")
            plt.imshow(img)

            plt.subplot(1,3,2)
            plt.title("Ground Truth")
            plt.imshow(gt_color)

            plt.subplot(1,3,3)
            plt.title("Prediction")
            plt.imshow(pr_color)

            plt.show()

            break


if __name__ == "__main__":
    visualize_seg()