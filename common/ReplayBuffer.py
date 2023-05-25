import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, obs_shape, action_space, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.state = np.zeros((max_size, *obs_shape))
        self.action = np.zeros((max_size, action_shape))
        self.next_state = np.zeros((max_size, *obs_shape))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.ind    = None
        self.device = device
        
    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state.cpu().numpy()
        self.action[self.ptr] = action.cpu().numpy()
        self.reward[self.ptr] = reward.cpu().numpy()
        self.next_state[self.ptr] = next_state.cpu().numpy()
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        self.ind = np.random.randint(0, self.size, size=batch_size)
                
        return (                
            torch.FloatTensor(self.state[self.ind]).to(self.device),
            torch.FloatTensor(self.action[self.ind]).to(self.device),
            torch.FloatTensor(self.reward[self.ind]).to(self.device),
            torch.FloatTensor(self.next_state[self.ind]).to(self.device),
            torch.FloatTensor(self.not_done[self.ind]).to(self.device),
        )

    def sample_state(self, batch_size):
        self.ind = np.random.randint(0, self.size, size=batch_size)
                
        return (                
            torch.FloatTensor(self.state[self.ind]).to(self.device)
        )

    def extract_backward(self, tag, num_steps):
        if tag == "state":
            return torch.FloatTensor(self.state[-num_steps: ]).to(self.device)
        elif tag == "action":
            return torch.FloatTensor(self.action[-num_steps: ]).to(self.device)
        elif tag == "reward":
            return torch.FloatTensor(self.reward[-num_steps: ]).to(self.device)
        elif tag == "next state":
            return torch.FloatTensor(self.next_state[-num_steps: ]).to(self.device)
        elif tag == "not done":
            return torch.FloatTensor(self.not_done[-num_steps: ]).to(self.device)
        else:
            raise ValueError
