from gym.spaces import Box
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from common.network import MLP_actor, MLP_critic, MLP_residual_net

class Policy():
    def __init__(self, obs_shape, action_space, hidden_size, latent_size, activation='tanh'):
        if isinstance(action_space, Box):
            self.action_space_low = torch.as_tensor(action_space.low)
            self.action_space_high = torch.as_tensor(action_space.high)

        self.action_space = action_space
        
        self.actor = MLP_actor(obs_shape, action_space, hidden_size, latent_size, activation)
        self.critic = MLP_critic(obs_shape, action_space, hidden_size, latent_size, activation)
        self.critic_target = deepcopy(self.critic)
        self.residual_net = MLP_residual_net(obs_shape, action_space, hidden_size, latent_size, activation)

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
        self.residual_net.to(device)

        if isinstance(self.action_space, Box):
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

        log_probs = dist.log_probs(sample)
        
        if isinstance(self.action_space, Box):
            return self.transform(sample), log_probs
        else:
            return sample, log_probs

    def get_q_dist(self, obs, actions):
        if len(obs.size()) == 5:
            squeezed, dims = True, obs.shape[:2]
            obs = obs.reshape(-1, *obs.shape[2:])
            actions = actions.reshape(-1, *actions.shape[2:])
        else:
            squeezed, dims = False, None

        q_dist = self.critic(obs, actions)

        return q_dist, {"squeezed": squeezed, "dims": dims}

    def get_q_value(self, obs, actions):
        q_dist, info = self.get_q_dist(obs, actions)
        qvalue = q_dist.mean
        if info["squeezed"]:        # then unsqueeze
            if len(qvalue.size()) == 2:
                qvalue = qvalue.reshape(*info["dims"], 1)
            elif len(qvalue.size()) == 3:
                qvalue = qvalue.reshape(*info["dims"], *qvalue.shape[1:])
            else:
                raise NotImplementedError

        return qvalue
        
    def get_residual(self, obs, actions):
        r = self.residual_net(obs, actions)
        return r

    def get_dist(self, obs):
        dist = self.actor(obs)

        return dist

    def sample_from_critic(self, obs, actions, critic_samples=1):
        q_dist, info = self.get_q_dist(obs, actions)

        samples = q_dist.sample((critic_samples, )) # ns x B x 1
        if len(samples.shape) == 3:
            samples = samples.transpose(1, 0)
            if info["squeezed"]:        # then unsqueeze
                if len(samples.size()) == 2:
                    samples = samples.reshape(*info["dims"], 1)
                elif len(samples.size()) == 3:
                    samples = samples.reshape(*info["dims"], *samples.shape[1:])
                else:
                    raise NotImplementedError
            return samples
        elif len(samples.shape) == 4:
            print("Sample shape == 4 happend for sample_from_critic")
            return samples.permute(1, 2, 0, 3)
        else:
            raise NotImplementedError

    def sample_action(self, obs, deterministic=False, rt=False):
        dist = self.actor(obs)
        
        if deterministic:
            sample = dist.mode()
        else:
            if rt:
                sample = dist.rsample()
            else:
                sample = dist.sample()

        log_probs = dist.log_probs(sample)
        
        if isinstance(self.action_space, Box):
            return self.transform(sample), log_probs
        else:
            return sample, log_probs

    def get_baseline(self, states, num_samples):
        size = len(states.size())
        if size == 5:
            state_dims = states.shape[:2]
            states = states.reshape(-1, *states.shape[2:]) # B x C x W x L


        dists = self.get_dist(states)            
        actions = dists.sample((num_samples, )) # ns x B x A
        action_dims = actions.shape[:2]
        actions = actions.reshape(-1, *actions.shape[2:])
        states = states.unsqueeze(0).repeat(num_samples, 1, 1, 1, 1) # ns x B x C x W x L
        states = states.reshape(-1, *states.shape[2:])
        qvalues = self.get_q_value(states, actions).detach() # ns x B x 1
        residuals = self.get_residual(states, actions)
        weights = 1 + residuals
        baselines = (weights * qvalues).reshape(*action_dims, 1).mean(0)

        if size == 5:
            baselines = baselines.reshape(*state_dims, 1)
        return baselines
    
    def evaluate_actions(self, obs, actions, samples=None):
        dist = self.actor(obs)
        
        if samples is None:
            if isinstance(self.action_space, Box):
                samples = self.draw_back(actions)
            else:
                samples = actions
        log_probs = dist.log_probs(samples)
        
        return log_probs
