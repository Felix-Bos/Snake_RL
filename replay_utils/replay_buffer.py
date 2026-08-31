import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, input_shape, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, *input_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *input_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.uint8)

    def store(self, state, action, reward, next_state, done):
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.tensor(self.states[idxs], device=self.device),
            torch.tensor(self.actions[idxs], device=self.device),
            torch.tensor(self.rewards[idxs], device=self.device),
            torch.tensor(self.next_states[idxs], device=self.device),
            torch.tensor(self.dones[idxs], device=self.device)
        )
    
    def __len__(self):
        return self.size