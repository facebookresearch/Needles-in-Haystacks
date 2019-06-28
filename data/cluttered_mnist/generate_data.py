# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# adapted from https://github.com/skaae/recurrent-spatial-transformer-code/blob/master/MNIST_SEQUENCE/create_mnist_sequence.py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import trange, tqdm
from scipy.ndimage import zoom
import scipy.misc
import sys
import pandas as pd


def create_sample(x, shape, num_distortions, distortions, scale_factor=1.,
                  distortion_scale=1., patch_size=0, centered=False, loc=-1):
    a, b = x.shape
    if patch_size == 0:
        if loc == -1:
            x_start = np.random.choice(shape - a)
            x_end = x_start + a
            y_start = np.random.choice(shape - b)
            y_end = y_start + b
        else:
            x_start = np.random.choice(np.minimum(loc, shape - a))
            x_end = x_start + a
            y_start = np.random.choice(np.minimum(loc, shape - b))
            y_end = y_start + b

        patch_map = None
    else:
        num_patches = shape // patch_size

        if centered:
            patch_x = patch_y = num_patches // 2
        else:
            patch_x = np.random.choice(num_patches)
            patch_y = np.random.choice(num_patches)
        x_start = patch_x * patch_size + np.random.choice(patch_size - a)
        x_end = x_start + a
        y_start = patch_y * patch_size + np.random.choice(patch_size - b)
        y_end = y_start + b

        patch_map = np.zeros((num_patches, num_patches))
        if np.sum(x) > 0:
            patch_map[patch_x, patch_y] = 1

    output = np.zeros((shape, shape))

    output[x_start:x_end, y_start:y_end] = x

    if num_distortions > 0:
        output = add_distortions(output, num_distortions, distortions, distortion_scale)

    if scale_factor != 1.:
        output = zoom(output, scale_factor)
    return output, patch_map


def add_distortions(canvas, num_distortions, distortions, distortion_scale=1.):
    out_shape = canvas.shape
    dist_size = distortions.shape[1:]
    for idx in np.random.choice(len(distortions), num_distortions):
        rand_distortion = distortions[idx]
        if distortion_scale != 1.:
            rand_distortion = zoom(rand_distortion, distortion_scale)
        dist_size = rand_distortion.shape
        rand_x = np.random.randint(out_shape[0]-dist_size[0])
        rand_y = np.random.randint(out_shape[1]-dist_size[1])
        canvas[rand_y:rand_y+dist_size[0],
               rand_x:rand_x+dist_size[1]] += rand_distortion

    return np.clip(canvas, 0., 1.)


def gen_detection_set(data, distortions, prefix, args):
    idx = (data.labels == args.digit)
    img_sel = data.images[idx]
    csv = pd.DataFrame(columns=['img_path', 'label'])

    for _ in range(args.duplicates):
        for img in tqdm(img_sel):
            imgp = os.path.join(args.out, '{}_{}.npz'.format(prefix, len(csv)))
            x, pm = create_sample(
                img.reshape((28, 28)), args.canvas, args.distortion_num,
                distortions, scale_factor=args.scale,
                distortion_scale=args.distortion_scale,
                patch_size=args.patch_size, centered=args.center_patch,
                loc=args.distortion_location)

            np.savez_compressed(imgp, x, pm)
            csv.loc[len(csv)] = [imgp, 1]

            imgp = os.path.join(args.out, '{}_{}.npz'.format(prefix, len(csv)))
            x, pm = create_sample(
                np.zeros((28, 28)), args.canvas, args.distortion_num + 1, distortions,
                scale_factor=args.scale, distortion_scale=args.distortion_scale,
                patch_size=args.patch_size, centered=args.center_patch,
                loc=args.distortion_location)

            np.savez_compressed(imgp, x, pm)
            csv.loc[len(csv)] = [imgp, 0]
    csv.to_csv(os.path.join(args.out, '{}.csv'.format(prefix)), index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate data for cluttered mnist')
    parser.add_argument('--out', '-o', help='directory to save images to')
    parser.add_argument('--mnist', '-m', help='path to mnist. will be downloaded if not present')
    parser.add_argument('--scale', '-s', default=1., type=float)
    parser.add_argument('--canvas', default=2048, type=int)
    parser.add_argument('--distortion-num', default=5000, type=int)
    parser.add_argument('--distortion-scale', default=1., type=float)
    parser.add_argument('--center-patch', default=False, action='store_true')
    parser.add_argument('--distortion-location', default=-1, type=int)
    parser.add_argument('--digit', default=3, type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--type', default='digits', type=str, choices=['digits', 'detection', 'test'])
    parser.add_argument('--patch-size', default=0, type=int)
    parser.add_argument('--duplicates', default=2, type=int)

    args = parser.parse_args()

    if args.patch_size > 0:
        assert args.patch_size > 28, 'patch size needs to be bigger than mnist digit'
        assert args.canvas % args.patch_size == 0, 'canvas needs to be multiple of patch size'

    os.makedirs(args.out, exist_ok=True)

    mnist = input_data.read_data_sets(args.mnist, one_hot=False)

    np.random.seed(1234)

    train_idx = (mnist.train.labels != args.digit)
    val_idx = (mnist.validation.labels != args.digit)
    test_idx = (mnist.test.labels != args.digit)

    train_distortions = mnist.train.images[train_idx].reshape([-1, 28, 28])
    val_distortions = mnist.validation.images[val_idx].reshape([-1, 28, 28])
    test_distortions = mnist.test.images[test_idx].reshape([-1, 28, 28])

    if args.type == 'test':
        idx = (mnist.train.labels == args.digit)
        img_sel = mnist.train.images[idx][0]
        print(args.digit)
        scipy.misc.imsave('test_orig.png', img_sel.reshape((28, 28)))
        test, _ = create_sample(img_sel.reshape((28, 28)), args.canvas,
                                args.distortion_num, train_distortions,
                                scale_factor=args.scale,
                                distortion_scale=args.distortion_scale,
                                patch_size=args.patch_size,
                                centered=args.center_patch,
                                loc=args.distortion_location)
        scipy.misc.imsave('test.png', test)
        sys.exit(0)
    elif args.type == 'detection':
        gen_detection_set(mnist.train, train_distortions, 'train', args)
        gen_detection_set(mnist.validation, val_distortions, 'val', args)
        gen_detection_set(mnist.test, test_distortions, 'test', args)

    with open(os.path.join(args.out, 'args'), 'w') as f:
        print(str(args), file=f)
