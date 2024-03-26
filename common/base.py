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

def CNNBase(num_inputs, hidden_size, latent_size, activation='relu'):
    act = activation_func(activation)
    hidden_features = nn.Sequential(
        nn.Conv2d(num_inputs, 32, 8, stride=4), act(),
        nn.Conv2d(32, 64, 4, stride=2), act(),
        nn.Conv2d(64, 32, 3, stride=1), act(), nn.Flatten(),
        nn.Linear(32 * 7 * 7, latent_size), act()
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

def create_base(obs_shape, hidden_size, activation, action_shape=None, latent_size=None):
    if len(obs_shape) == 3:
        if latent_size is None:
            latent_size = hidden_size
            print(f"latent_size is None")
        return CNNBase(obs_shape[0], hidden_size, latent_size, activation)
        
    elif len(obs_shape) == 1:
        if action_shape == None:
            return MLPBase(obs_shape[0], hidden_size, activation)
        else:
            return MLPBase(obs_shape[0] + action_shape, hidden_size, activation)
    else:
        raise NotImplementedError
