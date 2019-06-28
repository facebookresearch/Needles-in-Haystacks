# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import numpy as np
import torch
import random

from tqdm import tqdm
from torch.nn import functional as F


class NoRMSprop(torch.optim.RMSprop):
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

        return loss

def pred_stats(predictions, labels):
    TP = (predictions * labels).sum().cpu().item()
    FN = ((1 - predictions) * labels).sum().cpu().item()
    TN = ((1 - predictions) * (1 - labels)).sum().cpu().item()
    FP = (predictions * (1 - labels)).sum().cpu().item()

    return TP, TN, FP, FN

def precision_recall_accuracy(TP, TN, FP, FN):
    precision = 0. if TP == 0 else TP / (TP + FP)
    recall = 0. if TP == 0 else TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + TN + FN)

    return precision, recall, accuracy

def log_evaluation(epoch, statistics, writer, prefix):
    writer.add_scalar(prefix + '/loss', statistics['loss'], epoch)
    writer.add_scalar(prefix + '/precision', statistics['precision'], epoch)
    writer.add_scalar(prefix + '/recall', statistics['recall'], epoch)
    writer.add_scalar(prefix + '/accuracy', statistics['accuracy'], epoch)
    writer.add_scalar(prefix + '/grad_var_min', statistics['min_var'], epoch)
    writer.add_scalar(prefix + '/grad_var_mean', statistics['mean_var'], epoch)
    writer.add_scalar(prefix + '/grad_var_max', statistics['max_var'], epoch)
    writer.add_scalar(prefix + '/rmsprop_m2_min', statistics['rmsprop_m2'].min(), epoch)
    writer.add_scalar(prefix + '/rmsprop_m2_mean', statistics['rmsprop_m2'].mean(), epoch)
    writer.add_scalar(prefix + '/rmsprop_m2_max', statistics['rmsprop_m2'].max(), epoch)

    writer.add_histogram(prefix + '/probs', statistics['probs'], epoch)
    writer.add_histogram(prefix + '/grad_var', statistics['flat_var'], epoch)
    writer.add_histogram(prefix + '/rmsprop_m2', statistics['rmsprop_m2'], epoch)
    print('{} epoch {}. Loss {} Acc {}'.format(prefix, epoch, statistics['loss'], statistics['accuracy']))
    if 'all_accuracy' in statistics.keys():
        print('\tACC: {}\n\tPrec: {}\n\tRecall: {}'.format(statistics['all_accuracy'], statistics['all_precision'], statistics['all_recall']))

def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def get_rmsprop_m2(model, opt):
    # assumes single parameter group
    sq_avgs = [opt.state[p]['square_avg'].cpu().numpy().reshape(-1) for p in model.parameters() if p.requires_grad]
    return np.concatenate(sq_avgs, 0)

def evaluate(model, criterion, loader):
    model.eval()

    pseudo_opt = NoRMSprop(model.parameters())

    eval_stats = {}

    cur_grad_stats = [(0, torch.zeros_like(p), torch.zeros_like(p)) for p in model.parameters() if p.requires_grad] # count, mean, m2
    def update_grad_stats(grad_stats, grad):
        # see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        (count, mean, M2) = grad_stats
        count += 1
        delta = grad - mean
        mean += delta / count
        delta2 = grad - mean
        M2 += delta * delta2

        return (count, mean, M2)

    # Test the model
    correct = 0
    total = 0
    total_loss = 0
    n = 0
    all_probs = []

    tps = []
    tns = []
    fps = []
    fns = []

    for images, labels in tqdm(loader):
        model.zero_grad()
        pseudo_opt.zero_grad()
        if torch.cuda.is_available():
            if torch.cuda.device_count() >= 1:
                images = images.cuda()
            labels = labels.cuda()
        outputs, att = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        pseudo_opt.step()

        cur_grad_stats = [update_grad_stats(s, p.grad) for s, p in zip(cur_grad_stats, model.parameters()) if p.requires_grad]
        total_loss += loss.mean().cpu().item()

        predicted = torch.argmax(outputs.data, 1)
        TP, TN, FP, FN = pred_stats(predicted, labels)

        tps += [TP]
        tns += [TN]
        fps += [FP]
        fns += [FN]

        all_probs += list(F.softmax(outputs, 1).data.cpu().numpy())

        n += 1

    flat_var = np.concatenate([(m2.cpu().numpy() / c).reshape(-1) for c, _, m2 in cur_grad_stats], 0)

    precision, recall, accuracy = precision_recall_accuracy(
        np.sum(tps), np.sum(tns), np.sum(fps), np.sum(fns))
    eval_stats['loss'] = total_loss / n
    eval_stats['precision'] = precision
    eval_stats['recall'] = recall
    eval_stats['accuracy'] = accuracy
    eval_stats['probs'] = np.array(all_probs)
    eval_stats['flat_var'] = flat_var
    eval_stats['min_var'] = flat_var.min()
    eval_stats['mean_var'] = flat_var.mean()
    eval_stats['max_var'] = flat_var.max()
    eval_stats['rmsprop_m2'] = get_rmsprop_m2(model, pseudo_opt)

    return eval_stats
