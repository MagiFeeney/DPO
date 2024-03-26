import torch

class A2C():
    def __init__(self,
                 actor_critic,
                 num_mini_batch):

        self.actor_critic = actor_critic
        assert num_mini_batch == 1
        self.num_mini_batch = num_mini_batch

    def init_rollouts(self, rollouts):
        self.rollouts = rollouts
        
    def create_generator(self):
        data_generator = self.rollouts.feed_forward_generator(self.num_mini_batch) # only one batch, not shuffled

        for sample in data_generator:
            states, actions, advantages, _, returns = sample

            log_probs = self.actor_critic.evaluate_actions(states, actions)

            a2c_loss = -(advantages * log_probs).mean()

            yield {
                'data': a2c_loss,
                'type': 'loss',
                'batch': {
                    'states': states,
                    'actions': actions,
                    'returns': returns
                },
                'resample': False
            }
