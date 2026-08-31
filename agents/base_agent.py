import torch
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, n_actions, device, run_name):
        self.n_actions = n_actions
        self.device = device
        self.steps_done = 0
    
        self.writer = SummaryWriter(log_dir=f"runs/{run_name}")
        self.checkpoint_dir = "checkpoints"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def save(self, filename):
        """Sauvegarde générique du modèle 'online'."""
        path = os.path.join(self.checkpoint_dir, f"{filename}.pth")
        # On suppose que chaque agent a un attribut 'online_net'
        if hasattr(self, 'online_net'):
            torch.save(self.online_net.state_dict(), path)
            print(f"Modèle sauvegardé : {path}")
        else:
            print("Pas de 'online_net' à sauvegarder.")

    def load(self, filename):
        """Chargement générique."""
        if filename.endswith(".pth"):
            name = filename[:-4]
        else:
            name = filename

        candidate_path = filename
        if not os.path.isfile(candidate_path):
            candidate_path = os.path.join(self.checkpoint_dir, f"{name}.pth")

        path = candidate_path
        if os.path.exists(path):
            if hasattr(self, 'online_net'):
                self.online_net.load_state_dict(torch.load(path, map_location=self.device))
                print(f"Modèle chargé depuis : {path}")
                # Si l'agent a un réseau cible, on le synchronise
                if hasattr(self, 'target_net'):
                    self.target_net.load_state_dict(self.online_net.state_dict())
            else:
                print("L'agent n'a pas de 'online_net' à charger.")
        else:
            print(f"Aucun fichier trouvé : {path}")

    def log(self, name, value):
        """Wrapper pour Tensorboard."""
        self.writer.add_scalar(name, value, self.steps_done)

    @abstractmethod
    def select_action(self, obs, epsilon=None):
        """Retourne une action (int)"""
        pass

    @abstractmethod
    def store_transition(self, obs, action, reward, next_obs, done):
        """Stocke la transition (Buffer pour DQN, Rollout pour PPO)"""
        pass

    @abstractmethod
    def train_step(self):
        """
        Effectue une étape d'entraînement SI nécessaire.
        Retourne la loss (float) ou None.
        """
        pass

    def set_mode(self, mode="train"):
        """Passe le réseau en mode entraînement ou évaluation."""
        if hasattr(self, 'online_net'):
            if mode == "train":
                self.online_net.train()
            else:
                self.online_net.eval()