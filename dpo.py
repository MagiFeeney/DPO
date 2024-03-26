import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from common.utils import soft_update, hard_update, set_flat_params_to


class DPO():
    def __init__(self,
                 learner,
                 actor_critic,
                 dpo_epoch,
                 num_samples,
                 gamma,
                 tau,
                 alpha,
                 omega,
                 baseline_updates,
                 critic_updates,
                 UTD,
                 lr=None):

        self.learner = learner
        self.actor_critic = actor_critic
        self.set_alias(actor_critic)

        self.gamma = gamma
        self.tau   = tau
        self.dpo_epoch = dpo_epoch
        self.baseline_updates = baseline_updates
        self.num_samples    = num_samples
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)       
        self.baseline_optimizer = optim.Adam(self.residual_net.parameters(), lr=lr)

        self.omega = omega
        self.alpha = alpha

        self.critic_updates = critic_updates # TRPO critic updates
        
        self.UTD = UTD
        
    def set_alias(self, actor_critic):
        self.actor = actor_critic.actor
        self.critic = actor_critic.critic        
        self.critic_target = actor_critic.critic_target 
        self.residual_net = actor_critic.residual_net

    def update(self, rollouts, memory, batch_size):
        self.learner.init_rollouts(rollouts)

        for e in range(self.dpo_epoch):
            generator = self.learner.create_generator()

            for signal in generator:
                # Second flow
                states = memory.sample_state(batch_size=batch_size)
                actions, log_probs = self.actor_critic.sample_action(states, rt=False)
                qvalues = self.actor_critic.get_q_value(states, actions)
                baselines = self.actor_critic.get_baseline(states, self.num_samples)
                advantages = (qvalues - baselines).detach().clamp(0)

                projection_loss = (log_probs * (self.alpha * log_probs.detach() - advantages)).mean()

                if signal['type'] == 'loss':
                    # Interpolate two sources of loss
                    actor_loss = self.omega * signal['data'] + (1 - self.omega) * projection_loss
                elif signal['type'] == 'param':
                    set_flat_params_to(self.actor, signal['data']) # Have been scaled internally
                    # Interpolate second source
                    actor_loss = (1 - self.omega) * projection_loss

                # update the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                states_batch = signal['batch']['states']
                actions_batch = signal['batch']['actions']
                returns_batch = signal['batch']['returns']

                # minimize cross entropy
                if signal['resample']:
                    for i in range(self.critic_updates):
                        index = torch.randint(0, states_batch.size(0), (batch_size, ))
                        q_dists, _ = self.actor_critic.get_q_dist(states_batch[index], actions_batch[index])

                        critic_loss = -q_dists.log_probs(returns_batch[index].transpose(1, 0).unsqueeze(-1)).mean()

                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        self.critic_optimizer.step()
                else:
                    q_dists, _ = self.actor_critic.get_q_dist(states_batch, actions_batch)

                    critic_loss = -q_dists.log_probs(returns_batch.transpose(1, 0).unsqueeze(-1)).mean()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()
                
                soft_update(self.critic_target, self.critic, self.tau)

    def update_critic(self, memory, batch_size):
        for i in range(self.UTD):
            states, actions, rewards, next_states, not_dones = memory.sample(batch_size=batch_size)
            with torch.no_grad():
                next_actions, next_log_probs = self.actor_critic.sample_action(next_states)
                target_q_dists = self.critic_target(next_states, next_actions)
                means = rewards + not_dones * self.gamma * target_q_dists.mean
                sigmas = self.gamma * target_q_dists.stddev
                return_dists = Normal(means, sigmas)

            q_dists, _ = self.actor_critic.get_q_dist(states, actions)

            critic_loss = kl_divergence(return_dists, q_dists).mean()

            # update the critic
            self.critic_optimizer.zero_grad()        
            critic_loss.backward()
            self.critic_optimizer.step()

            soft_update(self.critic_target, self.critic, self.tau)

    def update_baseline(self, memory, batch_size):
        # baseline updates
        for i in range(self.baseline_updates):
            states, actions, _, _, _ = memory.sample(batch_size=batch_size)
            baselines = self.actor_critic.get_baseline(states, self.num_samples)
            qvalues = self.actor_critic.get_q_value(states, actions)

            baseline_loss = F.mse_loss(baselines, qvalues)

            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

