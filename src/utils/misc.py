import os
import sys
import time
import math
import shutil
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'load_ckpt2module',
    'load_ckpt',
    'save_checkpoint',
    'model_info',
    'write_logs',
    'prepare_directory',
    'AverageMeter'
]

def load_ckpt2module(ckpt, module, module_name):
    mdict = module.state_dict()
    to_be_removed = []
    for k, v in ckpt[module_name].items():
        if not k in mdict:
            to_be_removed.append(k)

    if len(to_be_removed) > 0:
        print ('Following items are removed from the ckpt["{}"]:'.format(module_name), end=' ')
        for i, k in enumerate(to_be_removed):
            if i == (len(to_be_removed) - 1):
                print (k)
            else:
                print (k, end=', ')
            del ckpt[module_name][k]
    else:
        print ('Nothing removed from the ckpt, great!')
  
    for duration in ('iteration', 'epoch'):
        if duration in ckpt:
            print ('Module in ckpt was trained {} {}s.'.format(ckpt[duration], duration))
      
    mdict.update(ckpt[module_name])
    module.load_state_dict(mdict)

def load_ckpt(path, device='cuda'):
    if device == 'cpu': ckpt = torch.load(path, map_location=lambda storage, loc: storage)
    else:               ckpt = torch.load(path)
    return ckpt

def save_checkpoint(state, dir, is_best=False, cur_iter=None):
    ckpt_file = os.path.join(dir, 'model.ckpt')
    torch.save(state, ckpt_file)
    if is_best:
        shutil.copyfile(ckpt_file, os.path.join(dir, 'model_best.ckpt'))
    if cur_iter is not None:
        shutil.copyfile(ckpt_file, os.path.join(dir, 'model_{:06d}.ckpt'.format(cur_iter)))

def model_info(model, model_name, save_dir=''):
    print ('Number of {} parameters: {}'.format(
        model_name,
        sum([p.data.nelement() for p in model.parameters()]))) 

    def save_model_desc(model, path):
        with open(path, 'w') as fid:
            fid.write(str(model))

    if save_dir:
        save_model_desc(
            model, os.path.join(save_dir, '{}_desc.txt'.format(model_name)))

def write_logs(FLAGS, project_dir=None):
    # save all setup into a log file
    _dict = vars(FLAGS)
    _list = sorted(_dict.keys())

    fid_setup = open(os.path.join(FLAGS.exp_dir, 'setup.txt'), 'w')
    for _k in _list:
        fid_setup.write('%s: %s\n' % (_k, _dict[_k]))
    fid_setup.flush()
    fid_setup.close()

    if project_dir is not None:
        for folder in ('src', 'scripts'):
            _from = os.path.join(project_dir, folder)
            _to = os.path.join(FLAGS.exp_dir, folder)
            if os.path.exists(_to): shutil.rmtree(_to)
            shutil.copytree(_from, _to)

def prepare_directory(directory, force_delete=False):
    if os.path.exists(directory) and not force_delete:
        print ('directory: %s already exists, backing up this folder ... ' % directory)
        backup_dir = directory + '_backup'

        if os.path.exists(backup_dir):
            print ('backup directory also exists, removing the backup directory first')
            shutil.rmtree(backup_dir, True)

        shutil.copytree(directory, backup_dir)

    shutil.rmtree(directory, True)

    os.makedirs(directory)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.__sum = 0
        self.__count = 0

    def update(self, val, n=1):
        self.val = val
        self.__sum += val * n
        self.__count += n

    @property
    def avg(self):
        if self.__count == 0:
            return 0.
        return self.__sum / self.__count