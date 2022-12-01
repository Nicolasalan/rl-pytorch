from collections import namedtuple, deque
import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # tamanho do buffer de repetição
BATCH_SIZE = 64         # tamanho do minilote

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
     """Buffer de tamanho fixo para armazenar tuplas de experiência."""

     def __init__(self, action_size, buffer_size, batch_size, seed):
          """Inicializar um objeto ReplayBuffer.

          parâmetros
          ======
               action_size (int): dimensão de cada ação
               buffer_size(int): tamanho máximo do buffer
               batch_size(int): tamanho de cada lote de treinamento
               semente (int): semente aleatória
          """
          self.action_size = action_size
          self.memory = deque(maxlen=buffer_size)  
          self.batch_size = batch_size
          self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
          self.seed = random.seed(seed)
     
     def add(self, state, action, reward, next_state, done):
          """Adicione uma nova experiência à memória."""
          experience = self.experience(state, action, reward, next_state, done)
          self.memory.append(experience)
     
     def sample(self):
          """Tente aleatoriamente um lote de experimentos de memória. """
          experiences = random.sample(self.memory, k=self.batch_size)

          states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
          actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
          rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
          next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
          dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
     
          return (states, actions, rewards, next_states, dones)

     def __len__(self):
          """Retorna o tamanho atual da memória interna."""
          return len(self.memory)