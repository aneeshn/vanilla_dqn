from collections import deque, namedtuple
import random
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer():
    """Fixed Length buffer to store the experience tuples"""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initialize a replay buffer

        :param action_size (int): dimensions of the action space
        :param buffer_size (int): the size of the buffer
        :param batch_size (int): size of the training batch
        :param seed (int):  the random seed
        :param device : the device where the code needs to run
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=['state','action','reward','next_state', 'done'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """
        Adding experince into the buffer

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """

        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """
        Randomly sample uniformly an experience from the queue
        :return:
        """

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)
