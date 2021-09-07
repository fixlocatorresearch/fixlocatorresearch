import random
from time import sleep

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn
from tqdm import tqdm
import model
import os.path as osp
import os
import numpy as np


class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.id = None

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth


def start_training(dataset, index_data):
    for i in range(len(index_data)):
        test_dataset = []
        train_dataset = []
        for j in range(len(dataset)):
            if j+1 in index_data[i]:
                test_dataset.append(dataset[j])
            else:
                train_dataset.append(dataset[j])
        model_ = model.FixLocator(h_size=64, feature_representation_size=128, drop_out_rate=0.5, layer_num=5,
                                  code_cover_len=1469)
        train(epochs=10, trainLoader=train_dataset, testLoader=test_dataset, model=model_, learning_rate=0.0001)


def evaluate_metrics(model, test_loader):
    model.eval()
    with torch.no_grad():
        hit = [0, 0, 0, 0, 0, 0]
        for data in tqdm(test_loader):
            correct = 0
            total = 0
            _, out = model(data)
            pred = out.argmax(dim=1)
            pred = pred.numpy()
            true_label = []
            for i in range(len(data.y[2])):
                true_label.append(data.y[2][i][0])
            for i in range(len(pred)):
                if pred[i] == 1 and true_label[i] == 1:
                    correct = correct + 1
                    total = total + 1
                if pred[i] == 1 and true_label[i] != 1:
                    total = total + 1
            if correct == 1:
                hit[0] = hit[0] + 1
            if correct == 2:
                hit[1] = hit[1] + 1
            if correct == 3:
                hit[2] = hit[2] + 1
            if correct == 4:
                hit[3] = hit[3] + 1
            if correct == 5:
                hit[4] = hit[4] + 1
            if correct > 5:
                hit[5] = hit[5] + 1
        print("Hit-1\t", hit[0])
        print("Hit-2\t", hit[1])
        print("Hit-3\t", hit[2])
        print("Hit-4\t", hit[3])
        print("Hit-5\t", hit[4])
        print("Hit-5+\t", hit[5])


def train(epochs, trainLoader, testLoader, model, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    try:
        for e in range(epochs):
            for index, _data in enumerate(tqdm(trainLoader, leave=False)):
                out = model(_data)
                y_1 = torch.reshape(torch.tensor(_data.y[1]), (-1,))
                y_2 = torch.reshape(torch.tensor(_data.y[2]), (-1,))
                out_1 = out[0].clone().detach().requires_grad_(True)
                out_2 = out[1].clone().detach().requires_grad_(True)
                loss = torch.autograd.Variable(criterion(out_1, y_1) + criterion(out_2, y_2), requires_grad = True)
                optimizer.zero_grad()  # if don't call zero_grad, the grad of each batch will be accumulated
                loss.backward()
                optimizer.step()
                sleep(0.05)
                if index % 20 == 0:
                    print('epoch: {}, batch: {}, loss: {}'.format(e + 1, index + 1, loss.data))
            evaluate_metrics(model=model, test_loader=testLoader)
            sleep(0.1)
    except KeyboardInterrupt:
        evaluate_metrics(model=model, test_loader=testLoader)


if __name__ == '__main__':
    dataset = []
    index_file = np.load(osp.join(os.getcwd(), 'processed/index.npy'))
    for i in range(len(index_file)):
        for j in range(len(index_file[i])):
            data = torch.load(osp.join(os.getcwd(), 'processed/data_{}.pt'.format(index_file[i][j])))
            dataset.append(data)
    start_training(dataset, index_file)
