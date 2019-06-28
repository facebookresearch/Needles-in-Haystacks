# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import numpy as np
import os
from tqdm import trange
import pandas as pd


def gen_set(num_pos, prefix, args):
    csv = pd.DataFrame(columns=['img_path', 'label'])

    for _ in trange(num_pos):
            imgp = os.path.join(args.out, '{}_{}.npz'.format(prefix, len(csv)))
            x = np.random.normal(size=(args.canvas, args.canvas))

            np.savez_compressed(imgp, x)
            csv.loc[len(csv)] = [imgp, 1]

            imgp = os.path.join(args.out, '{}_{}.npz'.format(prefix, len(csv)))
            x = np.random.normal(size=(args.canvas, args.canvas))

            np.savez_compressed(imgp, x)
            csv.loc[len(csv)] = [imgp, 0]
    csv.to_csv(os.path.join(args.out, '{}.csv'.format(prefix)), index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate data for noise experiments')
    parser.add_argument('--out', '-o', help='directory to save images to')
    parser.add_argument('--canvas', default=512, type=int)
    parser.add_argument('--num_train', default=11276, type=int)
    parser.add_argument('--num_val', default=500, type=int)
    parser.add_argument('--num_test', default=500, type=int)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    np.random.seed(1234)

    gen_set(args.num_train, 'train', args)
    gen_set(args.num_val, 'val', args)
    gen_set(args.num_test, 'test', args)

    with open(os.path.join(args.out, 'args'), 'w') as f:
        print(str(args), file=f)
