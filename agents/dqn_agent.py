import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

from agents.base_agent import BaseAgent
from networks_utils.qnet_factory import QnetFactory
from replay_utils.replay_buffer import ReplayBuffer

class DQNAgent(BaseAgent):
    def __init__(self, input_shape, n_actions, device, run_name, 
                 lr=1e-4, gamma=0.99, buffer_size=100000, batch_size=32,
                 model_config=None):
        """
        Agent DQN Standard.
        
        Args:
            input_shape (tuple): Forme de l'entrée (ex: (11,) ou (1, 40, 40))
            n_actions (int): Nombre d'actions possibles
            device (torch.device): CPU ou CUDA/MPS
            run_name (str): Nom pour les logs Tensorboard
            lr (float): Learning Rate
            gamma (float): Facteur d'actualisation (Discount factor)
            buffer_size (int): Taille de la mémoire
            batch_size (int): Taille du batch pour l'entrainement
            model_config (dict): Config des couches/activations pour la Factory
        """
        super().__init__(n_actions, device, run_name)
        
        self.gamma = gamma
        self.batch_size = batch_size

        # --- 1. Création des Réseaux via la Factory ---
        # Online Net : Celui qui apprend et qui joue
        self.online_net = QnetFactory.create_qnet(input_shape, n_actions, config=model_config).to(device)
        
        # Target Net : Celui qui sert de cible stable (copie du online)
        self.target_net = QnetFactory.create_qnet(input_shape, n_actions, config=model_config).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval() # Le target net n'est jamais entraîné directement

        # --- 2. Optimiseur ---
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        
        # --- 3. Mémoire ---
        self.memory = ReplayBuffer(buffer_size, input_shape, device)

    def select_action(self, obs, epsilon=0.0):
        """
        Sélectionne une action via Epsilon-Greedy.
        """
        # Incrémente le compteur de pas global (utile pour les logs)
        self.steps_done += 1
        
        # Exploration : Action aléatoire
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        
        # Exploitation : Action optimale selon le réseau
        with torch.no_grad():
            # Conversion numpy -> tensor et ajout dimension batch (1, ...)
            obs_t = torch.tensor(np.array([obs]), device=self.device)
            q_values = self.online_net(obs_t)
            
            # Retourne l'index de la plus grande Q-value
            return q_values.argmax().item()

    def store_transition(self, obs, action, reward, next_obs, done):
        """Stocke l'expérience dans le Replay Buffer"""
        self.memory.store(obs, action, reward, next_obs, done)

    def update_target_net(self):
        """Synchronise les poids du Target Net avec le Online Net"""
        self.target_net.load_state_dict(self.online_net.state_dict())
        # print("🔄 Target Network updated") # Décommente pour debug

    def train_step(self):
        """
        Une étape d'apprentissage DQN.
        Retourne la loss (float) si entraînement, sinon None.
        """
        # On n'apprend que si on a assez de données dans le buffer
        if len(self.memory) < self.batch_size:
            return None

        # 1. Échantillonnage d'un batch aléatoire
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 2. Calcul des Q-values actuelles : Q(s, a)
        # self.online_net(states) -> [Batch, n_actions]
        # .gather(1, actions) -> On ne garde que la Q-value de l'action qui a été prise
        q_eval = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 3. Calcul de la Cible (Target) : R + gamma * max(Q(s', a'))
        with torch.no_grad():
            # Le Target Net évalue les états suivants
            # .max(1)[0] prend la valeur max parmi les actions
            q_next = self.target_net(next_states).max(1)[0]
            
            # Formule de Bellman
            # Si done=1, le futur est 0, donc q_target = reward
            q_target = rewards + (self.gamma * q_next * (1 - dones))

        # 4. Calcul de la perte (Smooth L1 est souvent plus stable que MSE)
        loss = F.smooth_l1_loss(q_eval, q_target)
        
        # 5. Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clipping des gradients pour éviter l'explosion (stabilise l'entrainement)
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        
        self.optimizer.step()

        return loss.item()