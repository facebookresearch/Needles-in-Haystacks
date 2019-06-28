# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import torch
import argparse
import numpy as np
from data.cluttered_mnist.dataset import ClutteredMNISTDataset
from train import train


def get_datasets(data_path, data_scaling, data_balance, batch_size, num_examples):
    dataset = ClutteredMNISTDataset

    train_dataset = dataset(data_path, os.path.join(data_path, 'train.csv'),
                            data_scaling, num_examples, data_balance)

    weights = torch.Tensor(train_dataset.weights)
    # over and undersamples..
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=8, sampler=sampler, pin_memory=True)
    train_val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    val_dataset = dataset(data_path, os.path.join(data_path, 'val.csv'), data_scaling)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    test_dataset = dataset(data_path, os.path.join(data_path, 'test.csv'), data_scaling)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

    return train_loader, train_val_loader, val_loader, test_loader


def get_datasets_from_args(args):
    return get_datasets(
        args.data_path, args.data_scaling, args.data_balance, args.batch_size,
        args.num_examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluttered MNIST training script')
    parser.add_argument('--logdir', '-l', default='/tmp/cluttered_mnist')
    parser.add_argument('--batch_size', '-b', default=32, type=int)
    parser.add_argument('--num_epochs', '-e', default=2, type=int)
    parser.add_argument('--data_path', '-d', help='Path to data to be used.')
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--network_k', default=8, type=int)
    parser.add_argument('--network_att_type', default='mean_forward')
    parser.add_argument('--l2', default=0., type=float)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--kernel3', default=(1, 1, 0, 0),
                        help='How many blocks use 3x3 kernels. Input as 1_1_1_1 for (1, 1, 1, 1)')
    parser.add_argument('--network_width', default=0, type=int)
    parser.add_argument('--network_dropout', default=False, action='store_true')
    parser.add_argument('--test_all', default=False, action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--delayed_step', default=0, type=int)
    parser.add_argument('--data_scaling', default=1., type=float)
    parser.add_argument('--data_balance', default=0.5, type=float)
    parser.add_argument('--num_examples', default=11276, type=int)
    parser.add_argument('--opt', default='rmsprop', choices=['rmsprop', 'momentum', 'adam'])
    parser.add_argument('--norm', default='in_aff', choices=['in_aff', 'in', 'bn'])
    args, remaining_args = parser.parse_known_args()
    assert len(remaining_args) == 0

    if isinstance(args.kernel3, str):
        args.kernel3 = tuple(args.kernel3.split('_'))

    train_loader, train_val_loader, val_loader, test_loader = get_datasets_from_args(args)

    args.input_channels = 1
    args.shrinkage = np.maximum(1, train_loader.dataset.shrinkage)

    train(args, train_loader, train_val_loader, val_loader, test_loader)
