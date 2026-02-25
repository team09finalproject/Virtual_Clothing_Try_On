import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from networks import SegGenerator
from datasets import VITONDataset, VITONDataLoader


# ---- SAME Opt class used in training ----
class Opt:
    def __init__(self):
        self.dataset_dir = r"D:\Projects\virtual\dataset"
        self.dataset_mode = "train"
        self.dataset_list = "train_pairs.txt"

        self.load_height = 512
        self.load_width = 384

        self.batch_size = 1
        self.workers = 0
        self.shuffle = False

        self.semantic_nc = 13
        self.init_type = "xavier"
        self.init_variance = 0.02

        self.checkpoint_dir = "./checkpoints"


opt = Opt()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load dataset ----
dataset = VITONDataset(opt)
loader = VITONDataLoader(opt, dataset)

# ---- Create model ----
model = SegGenerator(opt, input_nc=20, output_nc=opt.semantic_nc).to(device)

# ---- Load trained weights ----
checkpoint = torch.load("checkpoints/seg_epoch_31.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

print("Model loaded successfully!")
# ---- Debug: Check what keys exist ----
for data in loader.data_loader:
    print("Available keys:", data.keys())
    break
# ---- Take one sample ----
sample = next(iter(loader.data_loader))
gt_mask = sample['parse_gt'].to(device)
parse_agnostic = sample['parse_agnostic'].to(device)
pose = sample['pose'].to(device)
c = sample['cloth']['unpaired'].to(device)
cm = sample['cloth_mask']['unpaired'].to(device)

# ---- Same preprocessing as training ----
parse_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')

seg_input = torch.cat((cm_down, c_masked_down, parse_down, pose_down), dim=1)

# ---- Prediction ----
import numpy as np

num_classes = 13
total_intersection = np.zeros(num_classes)
total_union = np.zeros(num_classes)

model.eval()

with torch.no_grad():
    for i, data in enumerate(loader.data_loader):

        if i == 50:
            break

        parse_agnostic = data['parse_agnostic'].to(device)
        pose = data['pose'].to(device)
        c = data['cloth']['unpaired'].to(device)
        cm = data['cloth_mask']['unpaired'].to(device)
        gt_mask = data['parse_gt'].to(device)

        gt_mask = F.interpolate(
            gt_mask.unsqueeze(1).float(),
            size=(256, 192),
            mode='nearest'
        ).squeeze(1).long()

        parse_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
        pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
        c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
        cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')

        seg_input = torch.cat((cm_down, c_masked_down, parse_down, pose_down), dim=1)

        output = model(seg_input)
        pred_mask = torch.argmax(output, dim=1)
        print(torch.unique(pred_mask))

        pred = pred_mask.cpu().numpy()
        
        gt = gt_mask.cpu().numpy()

        for cls in range(num_classes):
            pred_inds = (pred == cls)
            gt_inds = (gt == cls)

            intersection = (pred_inds & gt_inds).sum()
            union = (pred_inds | gt_inds).sum()

            total_intersection[cls] += intersection
            total_union[cls] += union

# Compute IoU
ious = []
for cls in range(num_classes):
    if total_union[cls] == 0:
        ious.append(np.nan)
    else:
        ious.append(total_intersection[cls] / total_union[cls])

for i, iou in enumerate(ious):
    print(f"Class {i} IoU: {iou:.4f}" if not np.isnan(iou) else f"Class {i} IoU: NaN")

mean_iou = np.nanmean(ious)
print("\nFinal Mean IoU over dataset:", mean_iou)



# ---- Plot ----
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Ground Truth")
plt.imshow(gt_mask[0].cpu().numpy())

plt.subplot(1,2,2)
plt.title("Predicted Mask")
plt.imshow(pred_mask[0].cpu().numpy())

plt.show()