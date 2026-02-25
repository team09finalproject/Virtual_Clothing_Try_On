class SegOptions:
    def __init__(self):
        self.dataset_dir = "C:\Users\Administrator\Desktop\virtual\viton-hd"
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


class GMMOptions:
    def __init__(self):
        self.dataset_dir = "D:/VITON-HD"
        self.dataset_mode = "train"
        self.dataset_list = "train_pairs.txt"

        self.load_height = 512
        self.load_width = 384

        self.grid_size = 5

        self.batch_size = 4
        self.workers = 4
        self.shuffle = True

        self.lr = 2e-4
        self.epochs = 70

        self.checkpoint_dir = "./checkpoints"