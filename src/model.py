# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class QNet(nn.Module):
#     """
#     Actor Model
#     """
#
#     def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
#         """
#         :param state_size (int): Dimensions of the state space
#         :param action_size (int): Dimensions of the action space
#         :param seed (int): Random seed
#         :param fc1_units (int): Number of nodes the first hidden fully connected layer
#         :param fc2_units (int): Number of the nodes second hidden fully connected layer
#         """
#         super(QNet, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_size)
#
#     def forward(self, state):
#         """Network mapping to values"""
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#
#         return x
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)