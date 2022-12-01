import numpy as np
import random
from collections import namedtuple, deque

from replaybuffer import ReplayBuffer
from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # tamanho do buffer de repetição
BATCH_SIZE = 64         # tamanho do minilote
GAMMA = 0.99            # factor de desconto
TAU = 1e-3              # para atualização suave dos parâmetros de destino
ALPHA = 5e-4            # Taxa de Aprendizagem
UPDATE_EVERY = 4        # com que frequência atualizar a rede

# utilizar GPU se disponível
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
     """Interagir e aprender com o meio ambiente."""

     def __init__(self, state_size, action_size, seed):
          """Inicializar um objeto Agente.
          
          parâmetros
          ======
               state_size (int): dimensão de cada estado
               action_size (int): dimensão de cada ação
               semente (int): semente aleatória
          """
          self.state_size = state_size
          self.action_size = action_size
          self.seed = random.seed(seed)

          # Q-Network
          self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device) # rede local para treinamento
          self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device) # rede alvo para treinamento
          self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=ALPHA) # otimizador para atualizar os parâmetros da rede local

          # Replay memory
          self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed) # buffer de repetição (acao, tamanho do buffer, tamanho do minilote, semente)
          # Inicialize a etapa de tempo (para atualizar todas as etapas)
          self.t_step = 0
     
     def step(self, state, action, reward, next_state, done):
          # Salvar experiência na memória de reprodução
          self.memory.add(state, action, reward, next_state, done) # (estado, ação, recompensa, próximo estado, terminado)
          
          # Aprenda todas as etapas ATUALIZADAS A CADA tempo.
          self.t_step = (self.t_step + 1) % UPDATE_EVERY
          if self.t_step == 0:
               # Se amostras suficientes estiverem disponíveis na memória, obtenha um subconjunto aleatório e aprenda
               if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample() # experiencia = array com (estado, ação, recompensa, próximo estado, terminado)
                    self.learn(experiences, GAMMA) # experincia com gamma 

     def act(self, state, eps=0.):
          """Retorna ações para um determinado estado de acordo com a política atual.
          
          parâmetros
          ======
               estado (array_like): estado atual
               eps (float): epsilon, para seleção de ação epsilon-gananciosa
          """
          state = torch.from_numpy(state).float().unsqueeze(0).to(device) # transformar o estado em tensor
          self.qnetwork_local.eval() # colocar a rede local em modo de avaliação
          with torch.no_grad(): # desativar o cálculo de gradiente
               action_values = self.qnetwork_local(state) # obter os valores de ação
          self.qnetwork_local.train() # colocar a rede local em modo de treinamento

          # Epsilon-greedy action selection
          if random.random() > eps: # se o número aleatório for maior que o epsilon
               return np.argmax(action_values.cpu().data.numpy()) # retornar a ação com maior valor
          else:
               return random.choice(np.arange(self.action_size)) # retornar uma ação aleatória

     def learn(self, experiences, gamma):
          """Atualize os parâmetros de valor usando determinado lote de tuplas experimentais.

          parâmetros
          ======
               experimentos (Tupla[torch.Variable]): tupla de (s, a, r, s', feito) tuplas
               gama (float): fator de desconto
           """
          states, actions, rewards, next_states, dones = experiences

          # Double DQN
          # next_local_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)

          # Obtenha os valores Q máximos previstos (para os próximos estados) do modelo de destino
          Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
          # Calcula Q alvos para os estados atuais
          # y = r + γ * maxQhat
          Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

          # Obtenha os valores Q esperados do modelo local
          # Q(\phi(s_t), a_j; \theta)
          Q_expected = self.qnetwork_local(states).gather(1, actions)

          # Perda de cálculo
          # execute gradient descent step at (y - Q)**2
          loss = F.mse_loss(Q_expected, Q_targets)
          # Minimizar a perda
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          # ------------------- update target network ------------------- #
          self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

     def soft_update(self, local_model, target_model, tau):
          """Parâmetros do modelo de atualização suave.
          θ_target = τ*θ_local + (1 - τ)*θ_target

          parâmetros
          ======
               local_model (modelo PyTorch): os pesos serão copiados de
               target_model (modelo PyTorch): os pesos serão copiados para
               tau (float): parâmetro de interpolação
          """
          # iter_params-
          for target_param, local_param in zip(target_model.parameters(), local_model.parameters()): # para cada parâmetro do modelo alvo e do modelo local
               tensor_aux = tau*local_param.data + (1.0-tau)*target_param.data # tensor auxiliar = tau * parâmetro local + (1 - tau) * parâmetro alvo
               target_param.data.copy_(tensor_aux) # copiar o tensor auxiliar para o parâmetro alvo