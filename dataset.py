import torch
from torch_geometric.data import Dataset
import os.path as osp


class fixlocatorData(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(fixlocatorData, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        list = []
        for i in range(30):
            exec("list.append('data_{}.pt')".format(i+1))
        return list

    def process(self):
        print("Data Incorrect")

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
