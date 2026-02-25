from options import SegOptions, GMMOptions
from train_seg import train_seg
from train_gmm import train_gmm

seg_opt = SegOptions()
gmm_opt = GMMOptions()

train_seg(seg_opt)
train_gmm(gmm_opt)