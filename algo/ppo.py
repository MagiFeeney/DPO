import torch

class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 num_mini_batch):

        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.num_mini_batch = num_mini_batch

    def init_rollouts(self, rollouts):
        self.rollouts = rollouts

    def create_generator(self):
        data_generator = self.rollouts.feed_forward_generator(self.num_mini_batch)

        for sample in data_generator:
            states, actions, advantages, old_log_probs, returns = sample

            log_probs = self.actor_critic.evaluate_actions(
                states, actions)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            clip_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surr2 = clip_ratio * advantages

            ppo_loss = -torch.min(surr1, surr2).mean()

            yield {
                'data': ppo_loss,
                'type': 'loss',
                'batch': {
                    'states': states,
                    'actions': actions,
                    'returns': returns
                },
                'resample': False
            }

