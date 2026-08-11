# -*- coding: utf-8 -*-
"""
# @time    : 18.11.2025 10:02
# @author  : zhouzy
# @file    : utils.py
"""
from torch.utils.data import Dataset
import torch
import numpy as np

def my_collate(batch):
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    return xs, ys

class ODFlowDataset(Dataset):
    def __init__(self, array_list, target_list, dest_sample_num = None):
        self.array_list = array_list
        self.target_list = target_list
        self.dest_sample_num = dest_sample_num

    def __len__(self):
        return len(self.array_list)

    def __getitem__(self, idx):
        m = self.array_list[idx].shape[1]
        if self.dest_sample_num == None:
            x = self.array_list[idx]
            x = torch.from_numpy(x).float()  # convert array to tensor
            #
            y = self.target_list[idx]
            y = torch.from_numpy(y).float()
        else:
            if m > self.dest_sample_num:
                sample_idx = np.random.choice(m, size=self.dest_sample_num, replace=False)

                x = self.array_list[idx][:, sample_idx, :]
                x = torch.from_numpy(x).float()  # convert array to tensor

                y = self.target_list[idx][:, sample_idx]
                y = torch.from_numpy(y).float()
            else:
                x = self.array_list[idx]
                x = torch.from_numpy(x).float()  # convert array to tensor
                #
                y = self.target_list[idx]
                y = torch.from_numpy(y).float()
        return x, y