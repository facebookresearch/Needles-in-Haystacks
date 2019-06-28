# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import numpy as np
from torch.utils.data.dataset import Dataset
import os
import random


class CamelyonImgDataset(Dataset):
    def __init__(self, pos_folder, neg_folder, ratio=1):
        pos_files = [os.path.join(pos_folder, f) for f in os.listdir(pos_folder) if 'seg' not in f and 'npy' in f]
        neg_files = [os.path.join(neg_folder, f) for f in os.listdir(neg_folder) if 'seg' not in f and 'npy' in f]

        len_pos = len(pos_files)
        len_neg = len_pos * ratio
        if len_neg > len(neg_files):
            len_neg = len(neg_files)
            len_pos = len_neg // ratio
            assert len_pos < len(pos_files)

        random.shuffle(pos_files)
        random.shuffle(neg_files)

        pos_files = pos_files[:len_pos]
        neg_files = neg_files[:len_neg]

        self.labels = np.array([1] * len(pos_files) + [0] * len(neg_files))
        self.data = np.array(pos_files + neg_files)

        order = np.arange(len(self.data))
        random.shuffle(order)

        self.labels = self.labels[order]
        self.data = self.data[order]

        print(len(self.data))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = np.load(self.data[index]).transpose(2, 0, 1).astype(np.float32)
        label = self.labels[index]
        return img, label
