# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import os
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
    mr_image = reader.open(os.path.join(args.data, img_name))
    ds = mr_image.getLevelDownsample(args.level)
    img_size = (np.array(mr_image.getDimensions()) / ds).astype(int)
    if img_size[0] < args.size or img_size[1] < args.size:
        return
    img = mr_image.getUCharPatch(0, 0, int(img_size[0]), int(img_size[1]), args.level)
    img = img.squeeze()
    patient_num = int(img_name.split('_')[1])
    center = patient_num // 20
    if center == 3:
        args.save_dir = os.path.join(args.save_dir, 'valid')
    if center == 4:
        args.save_dir = os.path.join(args.save_dir, 'test')

    orig_img_path = os.path.join(args.base_path, 'center_{}'.format(center), img_name)
    orig_img = reader.open(orig_img_path)
    orig_img_data = orig_img.getUCharPatch(0, 0, int(img_size[0]), int(img_size[1]), args.level)
    if img.sum() < args.low_roir * args.size * args.size:
        return
    labels, num = measure.label(img, return_num=True)
    for i_label in range(1, num + 1):
        valid_patch_counter = 0
        bin_mask = (labels == i_label)
        top, bottom, left, right =  get_bb_box(bin_mask)
        if bin_mask.sum() > args.size * args.size * args.high_roir:
            continue
        if None in (top, bottom, left, right):
            continue
        start_x = max(0, bottom - args.size)
        end_x = min(img.shape[0] - args.size, top)
        start_y = max(0, right - args.size)
        end_y = min(img.shape[1] - args.size, left)
        if end_x < start_x or end_y < start_y:
            continue
        if img[start_x:end_x, start_y:end_y].sum() < args.low_roir * args.size * args.size:
            continue
        for _ in range(10000):
            x = random.randint(start_x, end_x)
            y = random.randint(start_y, end_y)
            patch = img[x:x+args.size, y:y+args.size]
            if (patch.sum() / (args.size * args.size)) > args.low_roir and (patch.sum() / (args.size * args.size)) < args.high_roir and patch.sum() >= bin_mask.sum():
                orig_img_patch = orig_img_data[x:x+args.size, y:y+args.size]
                if not (orig_img_patch.sum(axis=-1) == 0).any():
                    save_path = os.path.join(args.save_dir, '{}_{}_{}_{}_img'.format(img_name.split('.')[0], args.level, i_label,
                                                                                valid_patch_counter))
                    np.save(save_path, orig_img_patch)
                    save_patch_path = os.path.join(args.save_dir,
                                                   '{}_{}_{}_{}_seg'.format(img_name.split('.')[0], args.level, i_label,
                                                                            valid_patch_counter))
                    np.save(save_patch_path, patch)
                    valid_patch_counter += 1
                    if valid_patch_counter >= args.k:
                        break


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Extract Crops")
    args.add_argument("--data", type=str, help='Path to CAMELYON17 dataset directory: training/lesion_annotations/ . This requires the annotations to be converted to .tiff already')
    args.add_argument("--base_path", type=str, help='Path to CAMELYON17 dataset directory: training')
    args.add_argument("--save_dir", type=str)
    args.add_argument("--low_roir", type=float, default=0.5)
    args.add_argument("--high_roir", type=float, default=1)
    args.add_argument("--size", type=int, default=128)
    args.add_argument("--level", type=int, default=3)
    args.add_argument("--n_workers", type=int, default=10)
    args.add_argument("--k", type=int, default=50)
    args.add_argument("--seed", type=int, default=42)
    args, remaining_args = args.parse_known_args()
    assert remaining_args == []

    args.save_dir = os.path.join(args.save_dir, 'level{}_size_{}_ratio_{}_{}'.format(args.level, args.size, args.low_roir, args.high_roir))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        os.makedirs(os.path.join(args.save_dir, 'valid'))
        os.makedirs(os.path.join(args.save_dir, 'test'))

    files = [f for f in sorted(os.listdir(args.data)) if '.tif' in f]
    futures = []
    with concurrent.futures.ProcessPoolExecutor(args.n_workers) as executor:
        for f in tqdm(files):
            futures.append(executor.submit(extract_img_crops, f, args))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass
