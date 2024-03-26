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
                                       None, device)
        self.eval_episodes = eval_episodes
        
    def eval(self):
        eval_episode_rewards = []
        state = self.eval_envs.reset()
        
        while len(eval_episode_rewards) < self.eval_episodes:
            with torch.no_grad():
                action, _ = self.actor_critic.act(state, deterministic=True)

            state, reward, done, infos = self.eval_envs.step(action)            
        
            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

        avg_reward = np.mean(eval_episode_rewards)

        print("Evaluation using {} episodes: mean reward {:.5f}\n".format(
            self.eval_episodes, avg_reward))

        return avg_reward 

