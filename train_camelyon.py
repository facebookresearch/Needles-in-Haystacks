# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import torch
import argparse

from data.camelyon.dataset import CamelyonImgDataset
from train import train


def get_datasets(pos_path, neg_path, batch_sz):
    train_dataset = CamelyonImgDataset(pos_path, neg_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_sz, shuffle=True, num_workers=8, pin_memory=True)
    train_val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_sz, shuffle=False, num_workers=8)

    val_dataset = CamelyonImgDataset(os.path.join(pos_path, 'valid'), os.path.join(neg_path, 'valid'))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_sz, shuffle=True, num_workers=8, pin_memory=True)

    test_dataset = CamelyonImgDataset(os.path.join(pos_path, 'test'), os.path.join(neg_path, 'test'))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_sz, num_workers=8, shuffle=False)

    return train_loader, train_val_loader, val_loader, test_loader


def get_datasets_from_args(args):
    return get_datasets(args.pos_path, args.neg_path, args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CAMELYON training script')
    parser.add_argument('--logdir', '-l', default='/tmp/camelyon')
    parser.add_argument('--batch_size', '-b', default=32, type=int)
    parser.add_argument('--num_epochs', '-e', default=20, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--network_k', default=8, type=int)
    parser.add_argument('--network_att_type', default='mean_forward')
    parser.add_argument('--l2', default=0., type=float)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--kernel3', default=(3, 4, 6, 3),
                        help='How many blocks use 3x3 kernels. Input as 1_1_1_1 for (1, 1, 1, 1)')
    parser.add_argument('--network_width', default=0, type=int)
    parser.add_argument('--network_dropout', default=False, action='store_true')
    parser.add_argument('--test_all', default=False, action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--delayed_step', default=0, type=int)
    parser.add_argument('--data_scaling', default=1., type=float)
    parser.add_argument('--data_balance', default=0.5, type=float)
    parser.add_argument('--opt', default='rmsprop', choices=['rmsprop', 'momentum', 'adam'])
    parser.add_argument('--norm', default='in_aff', choices=['in_aff', 'in', 'bn'])
    parser.add_argument('--pos_path', '-p', help='Path to positive examples')
    parser.add_argument('--neg_path', '-n', help='Path to negative examples')
    args, remaining_args = parser.parse_known_args()
    assert len(remaining_args) == 0

    if isinstance(args.kernel3, str):
        args.kernel3 = tuple(args.kernel3.split('_'))

    train_loader, train_val_loader, val_loader, test_loader = get_datasets_from_args(args)

    args.input_channels = 3
    args.shrinkage = 1

    train(args, train_loader, train_val_loader, val_loader, test_loader)
