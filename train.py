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
from tensorboardX import SummaryWriter

from model.model import Network

from utils import seed, evaluate, log_evaluation, pred_stats, precision_recall_accuracy, get_rmsprop_m2


def save_checkpoint(logdir, state):
    checkpoint_path = os.path.join(logdir, 'checkpoint.pth')
    tmp_checkpoint_path = os.path.join(logdir, 'tmp_checkpoint.pth')
    best_checkpoint_path = os.path.join(logdir, 'best_checkpoint.pth')

    print('Saving checkpoint')
    torch.save(state, tmp_checkpoint_path)
    if os.path.isfile(tmp_checkpoint_path):
        os.rename(tmp_checkpoint_path, checkpoint_path)

    if state['is_best']:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


def load_checkpoint(logdir, state):
    checkpoint_path = os.path.join(logdir, 'checkpoint.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        state.update(checkpoint)
        print('Loaded checkpoint')
        return True

    return False


def train(args, train_loader, train_val_loader, val_loader, test_loader):
    seed(args.seed)
    job_id = os.environ.get('SLURM_JOB_ID', 'local')

    print('Starting run {} with:\n{}'.format(job_id, args))

    writer = SummaryWriter(args.logdir)

    columns = ['epoch', 'eval_loss', 'eval_acc', 'eval_prec', 'eval_recall',
               'train_loss', 'train_acc', 'train_prec', 'train_recall',
               'test_loss', 'test_acc', 'test_prec', 'test_recall']
    stats_csv = pd.DataFrame(columns=columns)

    model = Network(
        k=args.network_k, att_type=args.network_att_type, kernel3=args.kernel3,
        width=args.network_width, dropout=args.network_dropout, compensate=True,
        norm=args.norm, inp_channels=args.input_channels)

    print(model)

    epochs = args.num_epochs * args.shrinkage
    milestones = np.array([80, 120, 160])
    milestones *= args.shrinkage
    milestones = list(milestones)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raw_model = model
    if torch.cuda.device_count() > 1:
        print('using multiple gpus')
        model = torch.nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    print(criterion)
    nn.utils.clip_grad_value_(raw_model.parameters(), 5.)
    if args.opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(raw_model.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.l2)
    elif args.opt == 'momentum':
        optimizer = torch.optim.SGD(raw_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(raw_model.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.l2)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

    state = {
        'epoch': 0,
        'step': 0,
        'state_dict': copy.deepcopy(raw_model.state_dict()),
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'lr_schedule': copy.deepcopy(lr_schedule.state_dict()),
        'best_acc': None,
        'best_epoch': 0,
        'is_best': False,
        'stats_csv': stats_csv,
        'config': vars(args)
    }

    if load_checkpoint(args.logdir, state):
        raw_model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        lr_schedule.load_state_dict(state['lr_schedule'])
        stats_csv = state['stats_csv']

    save_checkpoint(args.logdir, state)

    writer.add_text('args/str', str(args), state['epoch'])
    writer.add_text('job_id/str', job_id, state['epoch'])
    writer.add_text('model/str', str(model), state['epoch'])

    # Train the model
    for epoch in range(state['epoch'], epochs):
        lr_schedule.step()
        model.train()

        losses = []
        tps = []
        tns = []
        fps = []
        fns = []
        batch_labels = []
        delayed = 0
        writer.add_scalar('stats/lr', optimizer.param_groups[0]['lr'], epoch + 1)
        with tqdm(train_loader, desc="Epoch [{}/{}]".format(epoch+1, epochs)) as pbar:
            for images, labels in pbar:
                batch_labels += list(labels)
                if torch.cuda.is_available():
                    if torch.cuda.device_count() == 1:
                        images = images.cuda()
                    labels = labels.cuda()
                # Forward pass
                outputs, att = model(images)
                loss = criterion(outputs, labels)
                predicted = torch.argmax(outputs.data, 1)

                TP, TN, FP, FN = pred_stats(predicted, labels)
                cpu_loss = loss.mean().cpu().item()

                losses += [cpu_loss]
                tps += [TP]
                tns += [TN]
                fps += [FP]
                fns += [FN]
                # Backward and optimize
                delayed += 1
                if args.delayed_step > 0:
                    (loss / args.delayed_step).backward()
                else:
                    loss.backward()

                if args.delayed_step == 0 or (delayed + 1) % args.delayed_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    precision, recall, accuracy = precision_recall_accuracy(
                        np.sum(tps), np.sum(tns), np.sum(fps), np.sum(fns))

                    writer.add_scalar('train/loss', np.mean(losses), state['step'])
                    writer.add_scalar('train/precision', precision, state['step'])
                    writer.add_scalar('train/recall', recall, state['step'])
                    writer.add_scalar('train/accuracy', accuracy, state['step'])
                    writer.add_scalar('train/labels', np.mean(batch_labels), state['step'])
                    state['step'] += 1

                    delayed = 0
                    losses = []
                    tps = []
                    tns = []
                    fps = []
                    fns = []
                    batch_labels = []

                pbar.set_postfix(loss=cpu_loss)

        # step last backward if the step isn't done yet because of an 'incomplete'
        # delayed / accumulated batch
        if delayed > 0:
            optimizer.step()
            optimizer.zero_grad()

            precision, recall, accuracy = precision_recall_accuracy(
                np.sum(tps), np.sum(tns), np.sum(fps), np.sum(fns))

            writer.add_scalar('train/loss', np.mean(losses), state['step'])
            writer.add_scalar('train/precision', precision, state['step'])
            writer.add_scalar('train/recall', recall, state['step'])
            writer.add_scalar('train/accuracy', accuracy, state['step'])
            writer.add_scalar('train/labels', np.mean(batch_labels), state['step'])
            state['step'] += 1

        state['epoch'] = epoch + 1
        state['state_dict'] = copy.deepcopy(raw_model.state_dict())
        state['optimizer'] = copy.deepcopy(optimizer.state_dict())
        state['lr_schedule'] = copy.deepcopy(lr_schedule.state_dict())

        if args.opt == 'rmsprop':
            rms_m2 = get_rmsprop_m2(model, optimizer)
            writer.add_scalar('train/rmsprop_m2_min', rms_m2.min(), state['epoch'])
            writer.add_scalar('train/rmsprop_m2_mean', rms_m2.mean(), state['epoch'])
            writer.add_scalar('train/rmsprop_m2_max', rms_m2.max(), state['epoch'])
            writer.add_histogram('train/rmsprop_m2', rms_m2, state['epoch'])

        val_stats = evaluate(model, criterion, val_loader)
        log_evaluation(state['epoch'], val_stats, writer, 'eval')

        if state['best_acc'] is None or state['best_acc'] < val_stats['accuracy']:
            state['is_best'] = True
            state['best_acc'] = val_stats['accuracy']
            state['best_epoch'] = state['epoch']
        else:
            state['is_best'] = False

        if (state['is_best'] or state['epoch'] >= epochs or args.test_all):
            train_stats = evaluate(model, criterion, train_val_loader)
            log_evaluation(state['epoch'], train_stats, writer, 'train_eval')

            test_stats = evaluate(model, criterion, test_loader)
            log_evaluation(state['epoch'], test_stats, writer, 'test')

            stats_csv.loc[len(stats_csv)] = [
                state['epoch'], val_stats['loss'], val_stats['accuracy'],
                val_stats['precision'], val_stats['recall'],
                train_stats['loss'], train_stats['accuracy'],
                train_stats['precision'], train_stats['recall'],
                test_stats['loss'], test_stats['accuracy'],
                test_stats['precision'], test_stats['recall']]
        else:
            stats_csv.loc[len(stats_csv)] = [
                state['epoch'], val_stats['loss'], val_stats['accuracy'],
                val_stats['precision'], val_stats['recall'],
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan]

        save_checkpoint(args.logdir, state)

    writer.add_text('done/str', 'true', state['epoch'])

    print('done - stopping now')

    writer.close()
