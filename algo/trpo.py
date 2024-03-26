from copy import deepcopy
import torch
from common.OnTheFly import OnTheFly

class TRPO():
    def __init__(self,
                 actor_critic,
                 num_mini_batch,
                 omega,
                 max_kl=None,
                 damping=None):

        self.actor_critic = actor_critic
        
        self.num_mini_batch = num_mini_batch
        self.omega = omega
        self.max_kl = max_kl
        self.damping = damping

    def init_rollouts(self, rollouts):
        self.rollouts = rollouts

    def create_generator(self):
        data_generator = self.rollouts.feed_forward_generator(self.num_mini_batch)

        old_actor = deepcopy(self.actor_critic.actor)
        controller = OnTheFly(old_actor, self.actor_critic, \
                              self.omega, self.max_kl, self.damping)
        
        for sample in data_generator:
            states, actions, advs, old_log_probs, returns = sample
            
            new_params = controller.step(states, actions, advs, old_log_probs)

            yield {
                'data': new_params,
                'type': 'param',
                'batch': {
                    'states': states,
                    'actions': actions,
                    'returns': returns
                },
                'resample': True
            }


