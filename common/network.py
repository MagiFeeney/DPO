import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box, Discrete
from common.base import MLPBase, CNNBase, create_base
from common.distributions import DiagGaussian, Categorical, Beta

class MLP_actor(nn.Module):
    def __init__(self, obs_shape, action_space, hidden_size, activation='tanh'):
        super(MLP_actor, self).__init__()

        self.hidden = create_base(obs_shape, hidden_size, activation)
        if isinstance(action_space, Box):
            self.policy_head = Beta(hidden_size, action_space.shape[0])
        elif isinstance(action_space, Discrete):
            self.policy_head = Categorical(hidden_size, action_space.n)
        else:
            raise NotImplementedError

    def forward(self, obs):
        hidden_features = self.hidden(obs)
        return self.policy_head(hidden_features)

    
class MLP_residual_net(nn.Module):
    def __init__(self, obs_shape, action_space, hidden_size, activation='tanh'):
        super(MLP_residual_net, self).__init__()

        action_shape = get_action_shape(action_space)
        
        self.hidden = create_base(obs_shape, hidden_size, activation, action_shape)
        self.residual_linear = nn.Linear(hidden_size, 1)
        
    def forward(self, obs, actions):
        x = torch.cat([obs, actions], -1)
        hidden_features = self.hidden(x)
        return self.residual_linear(hidden_features)


class MLP_critic(nn.Module):
    def __init__(self, obs_shape, action_space, hidden_size, activation='tanh'):
        super(MLP_critic, self).__init__()

        action_shape = get_action_shape(action_space)        

        self.hidden = create_base(obs_shape, hidden_size, activation, action_shape)
        self.critic_head = DiagGaussian(hidden_size, 1)

    def forward(self, obs, actions):
        x = torch.cat([obs, actions], -1)        
        hidden_features = self.hidden(x)
        return self.critic_head(hidden_features)

    
def get_action_shape(action_space):
    if isinstance(action_space, Box):
        return action_space.shape[0]
    elif isinstance(action_space, Discrete):
        return action_space.n
    else:            
        raise NotImplementedError
