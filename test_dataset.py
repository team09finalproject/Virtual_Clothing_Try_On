from datasets import VITONDataset

class Opt:
    def __init__(self):
        self.dataset_dir = "C:/Users/Administrator/Desktop/virtual/viton-hd"
        self.dataset_mode = "train"
        self.dataset_list = "train_pairs.txt"
        self.load_height = 512
        self.load_width = 384
        self.semantic_nc = 13
        self.batch_size = 4
        self.workers = 4
        self.shuffle = True

opt = Opt()
dataset = VITONDataset(opt)

print("Dataset size:", len(dataset))