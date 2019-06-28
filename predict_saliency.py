# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import copy
import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from tensorboardX import SummaryWriter

from model.model import Network
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import imageio

from train_camelyon import get_datasets_from_args as cam_data
from train_mnist import get_datasets_from_args as mnist_data


def deploy(args, data_loader):
    model = Network(
        k=args.network_k, att_type=args.network_att_type, kernel3=args.kernel3,
        width=args.network_width, dropout=args.network_dropout, compensate=True,
        norm=args.norm, inp_channels=args.input_channels)

    print(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint_path = os.path.join(args.logdir, 'best_checkpoint.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise Exception('Couldnt load checkpoint.')

    df = pd.DataFrame(columns=['img', 'label', 'pred'])

    with tqdm(enumerate(data_loader)) as pbar:
        for i, (images, labels) in pbar:
            raw_label = labels
            raw_images = images
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            images.requires_grad = True
            # Forward pass
            outputs, att, localised = model(images, True)
            localised = F.softmax(localised.data, 3)[..., 1]
            predicted = torch.argmax(outputs.data, 1)
            saliency = torch.autograd.grad(outputs[:, 1].sum(), images)[0].data

            localised = localised[0].cpu().numpy()
            saliency = torch.sqrt((saliency[0] ** 2).mean(0)).cpu().numpy()
            raw_img = np.transpose(raw_images.numpy(), (0, 2, 3, 1)).squeeze()
            np.save(
                os.path.join(args.outpath, 'pred_{}.npy'.format(i)),
                localised)

            np.save(
                os.path.join(args.outpath, 'sal_{}.npy'.format(i)),
                saliency)

            df.loc[len(df)] = [i, raw_label.numpy().squeeze(), predicted.cpu().numpy().squeeze()]

    df.to_csv(os.path.join(args.outpath, 'pred.csv'), index=False)
    print('done - stopping now')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy script')
    parser.add_argument('--logdir', '-l', default='/tmp/model_logdir')
    # network args
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--network_k', default=8, type=int)
    parser.add_argument('--network_att_type', default='max_forward')
    parser.add_argument('--kernel3', default=(1, 1, 1, 1))
    parser.add_argument('--network_width', default=0, type=int)
    parser.add_argument('--network_dropout', default=False, action='store_true')
    parser.add_argument('--data_scaling', default=1., type=float)
    parser.add_argument('--norm', default='in_aff', choices=['in_aff', 'in', 'bn'])
    # data args
    parser.add_argument('--datatype', default='mnist', choices=['mnist', 'camelyon'])
    parser.add_argument('--data_path', '-d')
    parser.add_argument('--pos_path', '-p')
    parser.add_argument('--neg_path', '-n')
    args, remaining_args = parser.parse_known_args()
    assert len(remaining_args) == 0

    args.data_balance = 0.5
    args.batch_size = 1
    args.num_examples = 2
    args.outpath = os.path.join(args.logdir, 'saliency')

    if args.datatype == 'mnist':
        train_loader, train_val_loader, val_loader, test_loader = mnist_data(args)

        args.input_channels = 1
        args.shrinkage = np.maximum(1, train_loader.dataset.shrinkage)
    else:
        train_loader, train_val_loader, val_loader, test_loader = cam_data(args)

        args.input_channels = 3
        args.shrinkage = 1

    shutil.rmtree(args.outpath, ignore_errors=True)
    os.makedirs(args.outpath, exist_ok=True)

    deploy(args, val_loader)
