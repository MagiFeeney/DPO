import torch
import torch.nn as nn

def activation_func(act):
    if act == 'tanh':
        return nn.Tanh
    elif act == 'relu':
        return nn.ReLU
    elif act == 'elu':
        return nn.ELU
    elif act == 'silu':
        return nn.SiLU
    elif act == 'selu':
        return nn.SELU
    elif act == 'leakyrelu':
        return nn.LeakyReLU
    else:
        raise ValueError

def CNNBase(num_inputs, hidden_size, activation='relu'):
    act = activation_func(activation) 
    hidden_features = nn.Sequential(
        nn.Conv2d(num_inputs, 32, 8, stride=4), act(),
        nn.Conv2d(32, 64, 4, stride=2), act(),
        nn.Conv2d(64, 32, 3, stride=1), act(), Flatten(),
        nn.Linear(32 * 7 * 7, hidden_size), act()
    )

    return hidden_features


def MLPBase(num_inputs, hidden_size, activation='tanh'):
    act = activation_func(activation)
    hidden_features = nn.Sequential(
        nn.Linear(num_inputs, hidden_size),
        act(),
        nn.Linear(hidden_size, hidden_size),
        act()
    )
    
    return hidden_features

def create_base(obs_shape, hidden_size, activation, action_shape=None):
    if len(obs_shape) == 3:
        base = CNNBase
    elif len(obs_shape) == 1:
        base = MLPBase
    else:
        raise NotImplementedError

    if action_shape == None:
        return base(obs_shape[0], hidden_size, activation)
    else:
        return base(obs_shape[0] + action_shape, hidden_size, activation)
