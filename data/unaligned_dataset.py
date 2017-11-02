import os.path, glob
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
from pdb import set_trace as st
import random

class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.transform = get_transform(opt)

        datapath = os.path.join(opt.dataroot, opt.phase + '*')
        self.dirs = sorted(glob.glob(datapath))

        self.paths = [sorted(make_dataset(d)) for d in self.dirs]
        self.sizes = [len(p) for p in self.paths]

    def __getitem__(self, index):
        # Choose two of our domains to perform a pass on
        DA, DB = random.sample(range(len(self.dirs)), 2)

        index_A = random.randint(0, self.sizes[DA] - 1)
        index_B = random.randint(0, self.sizes[DB] - 1)

        A_path = self.paths[DA][index_A]
        B_path = self.paths[DB][index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        return {'A': A_img, 'B': B_img,
                'DA': DA, 'DB': DB,
                'path': A_path}

    def __len__(self):
        return max(self.sizes)

    def name(self):
        return 'UnalignedDataset'
