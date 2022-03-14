import numpy as np
import random
from collections import namedtuple, deque

from replaybuffer import ReplayBuffer
from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for smooth updating of target parameters
ALPHA = 5e-4            # Learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
     """Interact and learn from the environment."""

     def __init__(self, state_size, action_size, seed):
          """Initialize an Agent object.
          
          parameters
          ======
               state_size (int): dimension of each state
               action_size (int): dimension of each action
               seed (int): random seed
          """
          self.state_size = state_size
          self.action_size = action_size
          self.seed = random.seed(seed)

          # Q-Network
          self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
          self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
          self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=ALPHA)

          # Replay memory
          self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
          # Initialize the time step (to update all UPDATE_EVERY steps)
          self.t_step = 0
     
     def step(self, state, action, reward, next_state, done):
          # Save experience on playback memory
          self.memory.add(state, action, reward, next_state, done)
          
          # Learn all UPDATE_EVERY time steps.
          self.t_step = (self.t_step + 1) % UPDATE_EVERY
          if self.t_step == 0:
               # If enough samples are available in memory, get a random subset and learn
               if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

     def act(self, state, eps=0.):
          """Returns actions for a given state according to the current policy.
          
          parameters
          ======
               state (array_like): current state
               eps (float): epsilon, for epsilon-greedy action selection
          """
          state = torch.from_numpy(state).float().unsqueeze(0).to(device)
          self.qnetwork_local.eval()
          with torch.no_grad():
               action_values = self.qnetwork_local(state)
          self.qnetwork_local.train()

          # Epsilon-greedy action selection
          if random.random() > eps:
               return np.argmax(action_values.cpu().data.numpy())
          else:
               return random.choice(np.arange(self.action_size))

     def learn(self, experiences, gamma):
          """Update the value parameters using given batch of experiment tuples.

          parameters
          ======
               experiments (Tupla[torch.Variable]): tuple of (s, a, r, s', done) tuples
               gamma (float): discount factor
          """
          states, actions, rewards, next_states, dones = experiences

          # Double DQN
          # next_local_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)

          # Get the predicted maximum Q-values (for the next states) from the target model
          # is optimized for gpu use
          Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
          # Calculate Q targets for current states
          # y = r + γ * maxQhat
          Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

          # Get the expected Q-values from the local model
          # Q(\phi(s_t), a_j; \theta)
          Q_expected = self.qnetwork_local(states).gather(1, actions)

          # Loss of calculation
          # execute gradient descent step at (y - Q)**2
          loss = F.mse_loss(Q_expected, Q_targets)
         # Minimize the loss
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          # ------------------- update target network ------------------- #
          self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

     def soft_update(self, local_model, target_model, tau):
          """Soft update model parameters.
          θ_target = τ*θ_local + (1 - τ)*θ_target

          parameters
          ======
               local_model (PyTorch model): weights will be copied from
               target_model (PyTorch model): the weights will be copied to
               tau (float): interpolation parameter
          """
          # iter_params-
          for target_param, local_param in zip(target_model.parameters(), local_model.parameters()): 
               tensor_aux = tau*local_param.data + (1.0-tau)*target_param.data
               target_param.data.copy_(tensor_aux)