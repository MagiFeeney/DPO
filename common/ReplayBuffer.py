import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, obs_shape, action_space, num_processes, device, max_size=int(1e6)):
        self.max_size = max_size
        self.num_processes = num_processes
        self.ptr = 0
        self.size = 0

        self.max_steps = self.max_size // self.num_processes
        
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.state = np.zeros((self.max_steps, num_processes, *obs_shape))
        self.action = np.zeros((self.max_steps, num_processes, action_shape))
        self.next_state = np.zeros((self.max_steps, num_processes, *obs_shape))
        self.reward = np.zeros((self.max_steps, num_processes, 1))
        self.not_done = np.zeros((self.max_steps, num_processes, 1))

        self.device = device
        
    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state.cpu().numpy()
        self.action[self.ptr] = action.cpu().numpy()
        self.reward[self.ptr] = reward.cpu().numpy()
        self.next_state[self.ptr] = next_state.cpu().numpy()
        self.not_done[self.ptr] = 1. - done.reshape(-1, 1)

        self.ptr = (self.ptr + 1) % self.max_steps
        self.size = min(self.size + 1, self.max_steps)

    def hash_to_mat(self, arr):
        x = arr // self.num_processes
        y = arr % self.num_processes
        return tuple(x), tuple(y)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size * self.num_processes, size=batch_size)
        x, y = self.hash_to_mat(ind)
                
        return (                
            torch.FloatTensor(self.state[x, y]).to(self.device),
            torch.FloatTensor(self.action[x, y]).to(self.device),
            torch.FloatTensor(self.reward[x, y]).to(self.device),
            torch.FloatTensor(self.next_state[x, y]).to(self.device),
            torch.FloatTensor(self.not_done[x, y]).to(self.device)
        )

    def sample_state(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        x, y = self.hash_to_mat(ind)
        
        return (                
            torch.FloatTensor(self.state[x, y]).to(self.device)
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
