from replaybuffer import ReplayBuffer
from model import QNet

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with the environment and learns from it"""

    def __init__(self, state_size, action_size, seed):
        """

        :param state_size (int): dimension of each state
        :param action_size (int): dimension of each action
        :param seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

        # Definition for the Q-network as well as the target network
        self.qnetwork_local = QNet(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNet(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = LR)

        #Replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy

        :param state:
        :param ep:
        :return:
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon Greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples

        :param experiences:
        :param gamma:
        :return:
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values for next states from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets_next = Q_targets_next.double()

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (~dones))
        Q_targets = Q_targets.double()

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        Q_expected = Q_expected.double()

        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update the target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model
        :param local_model:
        :param target_model:
        :param tau:
        :return:
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
