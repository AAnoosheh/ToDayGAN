import os.path, glob
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.transform = get_transform(opt)

        datapath = os.path.join(opt.dataroot, opt.phase + '*')
        self.dirs = sorted(glob.glob(datapath))

        self.paths = [sorted(make_dataset(d)) for d in self.dirs]
        self.sizes = [len(p) for p in self.paths]

    def load_image(self, dom, idx):
        path = self.paths[dom][idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, path

    def __getitem__(self, index):
        if not self.opt.isTrain:
            # Choice of domain order sequential during test
            DA = index % len(self.dirs)
            # Choice of image within domain not always ordered
            if self.opt.serial_test:
                index_A = index // len(self.dirs)
            else:
                index_A = random.randint(0, self.sizes[DA] - 1)
        else:
            # Choose two of our domains to perform a pass on
            DA, DB = random.sample(range(len(self.dirs)), 2)
            index_A = random.randint(0, self.sizes[DA] - 1)

        A_img, A_path = self.load_image(DA, index_A)
        bundle = {'A': A_img, 'DA': DA, 'path': A_path}

        if self.opt.isTrain:
            index_B = random.randint(0, self.sizes[DB] - 1)
            B_img, _ = self.load_image(DB, index_B)
            bundle.update( {'B': B_img, 'DB': DB} )

        return bundle

    def __len__(self):
        return max(self.sizes)

    def name(self):
        return 'UnalignedDataset'
