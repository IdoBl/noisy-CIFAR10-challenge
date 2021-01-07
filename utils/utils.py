"""Utility module

Includes helper classes and functions for the training process
"""

import numpy as np
import logging
from torch.optim import SGD, Adam, RMSprop, Adagrad
# This import is for the dynamic loading of the different models
from models import *
import sys


def adjust_weight_decay(model, l2_value):
    """Weight decay using L2 regularization. Used by the model SGD optimizer"""

    conv, fc = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            # frozen weights
            continue
        if 'module.fc1' in name:
            fc.append(param)
        else:
            conv.append(param)
    params = [{'params': conv, 'weight_decay': l2_value}, {'params': fc, 'weight_decay': 0.01}]
    return params


def get_optimizer(optim_name, fixed_cnn, args):
    """Return the asked optimizer"""

    optimizers = {
        'sgd': SGD(params=adjust_weight_decay(fixed_cnn, args.l2_reg), lr=args.lr, momentum=0.9, nesterov=True),
        'adam': Adam(params=adjust_weight_decay(fixed_cnn, args.l2_reg), lr=args.lr),
        'adagrad': Adagrad(params=adjust_weight_decay(fixed_cnn, args.l2_reg), lr=args.lr),
        'rmsprop': RMSprop(params=adjust_weight_decay(fixed_cnn, args.l2_reg), lr=args.lr)
    }
    return optimizers.get(optim_name)


def load_model(model_name):
    """Dynamically load the requested model"""
    return getattr(sys.modules[__name__], model_name)()


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def log_display(epoch, global_step, time_elapse, **kwargs):
    display = 'epoch=' + str(epoch) + \
              '\tglobal_step=' + str(global_step)
    for key, value in kwargs.items():
        display += '\t' + str(key) + '=%.5f' % value
    display += '\ttime=%.2fit/s' % (1. / time_elapse)
    return display


def accuracy(output, target, topk=(1,)):
    """Calculate the accuracy for the asked results"""
    maxk = max(topk)

    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(1/batch_size))
    return res


def count_parameters_in_MB(model):
    """Calculate capacity of model parameters in MB for debugging/info purpose"""
    return sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary_head" not in name)/1e6


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)
