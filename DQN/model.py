import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor Model (Policy)"""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=32):
            """Initialize the parameters and build the model.
            parameters
            ======
                  state_size: size of state space.
                  action_size: action space size.
                  seed (int): random seed
                  fc1_units (int): Number of nodes in the first hidden layer
                  fc2_units (int): Number of nodes in the second hidden layer
            """
            super(QNetwork, self).__init__()
            # Initialize parameters
            self.seed = torch.manual_seed(seed)
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
            """Build a network that maps state -> action values."""
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            return self.fc3(x)