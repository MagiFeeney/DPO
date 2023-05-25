import numpy as np
import torch
from common import utils
from common.envs import make_vec_envs


class Evaluator():
    def __init__(self,
                 actor_critic,
                 env_name,
                 seed,
                 num_processes,
                 device,
                 eval_episodes=10):

        self.actor_critic = actor_critic
        self.eval_envs = make_vec_envs(env_name, seed + 100, num_processes,
                                       None, None, device, False)
        self.eval_episodes = eval_episodes
        
    def eval(self, obs_rms):
        vec_norm = utils.get_vec_normalize(self.eval_envs)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.obs_rms = obs_rms
        
        avg_reward = 0
        for i in range(self.eval_episodes):
            state, done = self.eval_envs.reset(), False
            while not done:
                with torch.no_grad():        
                    action, _ = self.actor_critic.act(state, deterministic=True)

                # transition infomation
                state, reward, done, _ = self.eval_envs.step(action)
                avg_reward += reward

        avg_reward /= self.eval_episodes

        print("Evaluation using {} episodes: mean reward {:.5f}\n".format(
            self.eval_episodes, avg_reward.item()))

        return avg_reward 

