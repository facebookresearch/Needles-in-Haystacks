# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import numpy as np
import os
import pandas as pd
import numpy as np
from skimage import measure
import random
from tqdm import tqdm
import concurrent.futures

import sys
# NOTE: uncomment the following line to add ASAP to the PYTHONPATH or
# add it manually with the correct location
# sys.path.append('/opt/ASAP/bin')

import multiresolutionimageinterface as mir


def get_bb_box(patch):
    top, bottom, left, right = None, None, None, None
    for i in range(len(patch)):
        if np.any(patch[i]):
            if top is None:
                top = i
            else:
                bottom = i
    for i in range(len(patch[0])):
        if np.any(patch[:, i]):
            if left is None:
                left = i
            else:
                right = i
    return top, bottom, left, right


def extract_img_crops(img_name, args):

    np.random.seed(args.seed)
    random.seed(args.seed)

    reader = mir.MultiResolutionImageReader()
    patient_num = int(img_name.split('_')[1])
    center = patient_num // 20
    if center == 3:
        args.save_dir = os.path.join(args.save_dir, 'valid')
    if center == 4:
        args.save_dir = os.path.join(args.save_dir, 'test')
    orig_img_path = os.path.join(args.base_path, 'center_{}'.format(center), img_name)
    orig_img = reader.open(orig_img_path)
    ds = orig_img.getLevelDownsample(args.level)
    img_size = (np.array(orig_img.getDimensions()) / ds).astype(int)
    if img_size[0] < args.size or img_size[1] < args.size:
        return
    orig_img_data = orig_img.getUCharPatch(0, 0, int(img_size[0]), int(img_size[1]), args.level)
    valid_patch_counter = 0
    for _ in range(10000):
        x_i = random.randint(0, img_size[1] - args.size)
        y_i = random.randint(0, img_size[0] - args.size)
        orig_img_patch = orig_img_data[x_i:x_i + args.size, y_i:y_i + args.size]
        if (orig_img_patch.sum(axis=-1) == 0).any() or orig_img_patch.mean(axis=(0, 1))[1] > 200:
            continue
        save_path = os.path.join(args.save_dir, '{}_{}_{}'.format(img_name.split('.')[0], args.level, valid_patch_counter))
        np.save(save_path, orig_img_patch)
        valid_patch_counter += 1
        if valid_patch_counter >= args.k:
            break


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Extract Crops")
    args.add_argument("--label_path", type=str, help='Path to CAMELYON17 dataset file: training/stage_labels.csv')
    args.add_argument("--base_path", type=str, help='Path to CAMELYON17 dataset directory: training/')
    args.add_argument("--save_dir", type=str)
    args.add_argument("--size", type=int, default=128)
    args.add_argument("--level", type=int, default=3)
    args.add_argument("--k", type=int, default=100)
    args.add_argument("--seed", type=int, default=42)
    args, remaining_args = args.parse_known_args()
    assert remaining_args == []

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        os.makedirs(os.path.join(args.save_dir, 'valid'))
        os.makedirs(os.path.join(args.save_dir, 'test'))

    df = pd.read_csv(args.label_path)
    files = df[df['stage'] == 'negative']['patient'].values
    futures = []
    with concurrent.futures.ProcessPoolExecutor(10) as executor:
        for f in tqdm(files):
            futures.append(executor.submit(extract_img_crops, f, args))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass
