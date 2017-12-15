import torch.utils.data


class DataLoader():
    def name(self):
        return 'DataLoader'

    def __init__(self, opt):
        self.opt = opt
        self.dataset = UnalignedDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
