from main import demo_work
import numpy as np
import os.path as osp
import torch
import os
from main import Tree


a = np.load("D:\\research\\Fault_Localization\\fixlocatorcode\\processed\\index.npy", allow_pickle=True)
b = np.array([a[0]])
np.save("index.npy", b)
dataset = []
index_file = np.load(osp.join(os.getcwd(), 'processed/index.npy', ), allow_pickle=True)
for i in range(len(index_file)):
    for j in range(len(index_file[i])):
        data = torch.load(osp.join(os.getcwd(), 'processed/data_{}.pt'.format(index_file[i][j])))
        dataset.append(data)
demo_work(dataset, index_file)

