import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Modelo de Ator (Política)"""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=32):
                """Inicializar os parâmetros e construir o modelo.
                 parâmetros
                 ======
                         state_size: tamanho do espaço de estado.
                         action_size: tamanho do espaço de ação.
                         semente (int): semente aleatória
                         fc1_units (int): Número de nós na primeira camada oculta
                         fc2_units (int): Número de nós na segunda camada oculta
                 """
                super(QNetwork, self).__init__()
                self.seed = torch.manual_seed(seed) # define a semente aleatória manualmente
                self.fc1 = nn.Linear(state_size, fc1_units) # camada de entrda com 128 nos
                self.fc2 = nn.Linear(fc1_units, fc2_units) # camada oculta com 128 de entrada e 32 de saida
                self.fc3 = nn.Linear(fc2_units, action_size) # camada de saida com acoes possiveis

    def forward(self, state):
                """Construir uma rede que mapeia estado -> valores de ação."""
                x = F.relu(self.fc1(state)) # funcao de ativação relu
                x = F.relu(self.fc2(x)) # funcao de ativação relu
                return self.fc3(x) # retorna a camada de saida