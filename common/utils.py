import glob
import os
import numpy as np
import torch
import torch.nn as nn
from common.envs import VecNormalize


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None

def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)            

def set_flat_params_to(model, new_params):
    prev_ind = 0
    for param in model.parameters():
        n_params = param.numel()
        param.data.copy_(
            new_params[prev_ind: prev_ind + n_params].view(param.size())
        )
        prev_ind += n_params
