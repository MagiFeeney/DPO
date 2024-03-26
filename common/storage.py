import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class RolloutStorage(object):
    def __init__(self,
                 num_steps,
                 num_processes,
                 obs_shape,
                 action_space,
                 critic_samples):
        
        self.states = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.log_probs = torch.zeros(num_steps + 1, num_processes, 1)
        self.advantages = torch.zeros(num_steps, num_processes, 1)
        
        self.qvalues = torch.zeros(num_steps + 1, num_processes, critic_samples, 1)
        self.particles = torch.zeros(num_steps, num_processes, critic_samples, 1) # particles of advantages
        self.returns = torch.zeros(num_steps, num_processes, critic_samples, 1)
        
        self.rewards = torch.zeros(num_steps, num_processes, 1, 1)
        self.baselines = torch.zeros(num_steps, num_processes, 1, 1)
        
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1, 1)
        self.truncated = torch.zeros(num_steps, num_processes)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.states = self.states.to(device)
        self.rewards = self.rewards.to(device)
        self.qvalues = self.qvalues.to(device)
        self.particles = self.particles.to(device)
        self.advantages = self.advantages.to(device)
        self.returns = self.returns.to(device)
        self.log_probs = self.log_probs.to(device)
        self.baselines = self.baselines.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.truncated = self.truncated.to(device)

    def insert(self, states, actions, log_probs,
               rewards, masks, truncated):
        self.states[self.step + 1].copy_(states)
        self.actions[self.step].copy_(actions)
        self.log_probs[self.step].copy_(log_probs)
        self.rewards[self.step].copy_(rewards.unsqueeze(-1))
        self.masks[self.step + 1].copy_(masks.unsqueeze(-1))
        self.truncated[self.step].copy_(truncated)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_advantages(self,
                           gamma,
                           uae_lambda,
                           normalize=True,
                           nu=None):
        uae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + \
                gamma * self.qvalues[step + 1] * self.masks[step + 1] - \
                self.baselines[step]
            y = self.qvalues[step] - self.baselines[step]
            discounted_uae = gamma * uae_lambda * self.masks[step + 1] * uae
            self.particles[step] = delta + discounted_uae
            uae = (delta - y) + discounted_uae

        if nu is not None:
            adv_q = self.qvalues[:-1] - self.baselines
            self.particles = nu * adv_q + (1 - nu) * self.particles
            self.advantages = self.particles.mean(-2)
        else:
            self.advantages = self.particles.mean(-2)

        # estimate the lambda returns for all particles
        self.returns.copy_(self.particles + self.baselines)
        
        if normalize:
            self.advantages = (self.advantages - self.advantages.mean(0)) / (
                self.advantages.std(0) + 1e-8) # statistics along the dimension of processes

                    
    def feed_forward_generator(self,
                               num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if num_mini_batch == 1:
            sampler = [list(range(batch_size))] # No shuffle, whole batch
        else:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
            sampler = BatchSampler(
                SubsetRandomSampler(range(batch_size)),
                mini_batch_size,
                drop_last=True)

        for indices in sampler:
            states_batch = self.states[:-1].view(-1, *self.states.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            advs_batch = self.advantages.view(-1, 1)[indices]
            old_log_probs_batch = self.log_probs[:-1].view(-1, 1)[indices]
            returns_batch = self.returns.view(-1, self.returns.size(-2))[indices]

            yield states_batch, actions_batch, advs_batch, old_log_probs_batch, returns_batch
