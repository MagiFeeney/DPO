import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from common.network import MLP_actor, MLP_critic, MLP_residual_net

class Policy():
    def __init__(self, obs_shape, action_space, hidden_size, activation='tanh'):
        self.action_space_low = torch.as_tensor(action_space.low)
        self.action_space_high = torch.as_tensor(action_space.high)
        
        self.actor = MLP_actor(obs_shape, action_space, hidden_size, activation)
        self.critic = MLP_critic(obs_shape, action_space, hidden_size, activation)
        self.critic_target = deepcopy(self.critic)
        self.residual_net = MLP_residual_net(obs_shape, action_space, hidden_size, activation)

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
        self.residual_net.to(device)
        
        self.action_space_low = self.action_space_low.to(device)
        self.action_space_high = self.action_space_high.to(device)
        
    def transform(self, sample):
        action = sample * (self.action_space_high - \
                           self.action_space_low) + self.action_space_low

        return action

    def draw_back(self, action):
        sample = (action - self.action_space_low) / \
            (self.action_space_high - self.action_space_low)
            
        return sample.clamp(1e-8) # avoid boundary effect due to transformation
        
    def act(self, obs, deterministic=False):
        dist = self.actor(obs)

        if deterministic:
            sample = dist.mode()
        else:
            sample = dist.sample()

        action = self.transform(sample)
        log_probs = dist.log_probs(sample)

        return action, log_probs

    def get_q_dist(self, obs, actions):
        q_dist = self.critic(obs, actions)

        return q_dist

    def get_q_value(self, obs, actions):
        q_dist = self.critic(obs, actions)

        return q_dist.mean
        
    def get_residual(self, obs, actions):
        r = self.residual_net(obs, actions)
        return r

    def get_dist(self, obs):
        dist = self.actor(obs)

        return dist

    def sample_from_critic(self, obs, actions, critic_samples=1):
        q_dist = self.get_q_dist(obs, actions)
        
        samples = q_dist.sample((critic_samples, )).transpose(1, 0)
        
        return samples

    def sample_action(self, obs, deterministic=False, rt=False):
        dist = self.actor(obs)
        
        if deterministic:
            sample = dist.mode()
        else:
            if rt:
                sample = dist.rsample()
            else:
                sample = dist.sample()

        action = self.transform(sample)
        log_probs = dist.log_probs(sample) 
        
        return action, log_probs

    def get_baseline(self, states, num_samples):

        dists = self.get_dist(states)
        actions = dists.sample((num_samples, )).transpose(1, 0)
        states = states.unsqueeze(1).repeat(1, num_samples, 1)
        qvalues = self.get_q_value(states, actions).detach()
        residuals = self.get_residual(states, actions)
        weights = 1 + residuals
        baselines = (weights * qvalues).mean(1)

        return baselines
    
    def evaluate_actions(self, obs, actions, samples=None):
        dist = self.actor(obs)
        
        if samples is None:
            samples = self.draw_back(actions)
        log_probs = dist.log_probs(samples)
        
        return log_probs
