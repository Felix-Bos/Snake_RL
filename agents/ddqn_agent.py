import torch
import torch.nn.functional as F
from agents.dqn_agent import DQNAgent

class DDQNAgent(DQNAgent):
    def train_step(self):
        # On n'apprend que si on a assez de données
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 1. Q(s, a) (Comme DQN)
        q_eval = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 2. Q Target (Spécifique DDQN)
        with torch.no_grad():
            # A. Le réseau Online CHOISIT la meilleure action pour l'état suivant
            next_actions = self.online_net(next_states).argmax(1)
            
            # B. Le réseau Target ÉVALUE la valeur de cette action
            q_next = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # C. Formule de Bellman
            q_target = rewards + (self.gamma * q_next * (1 - dones))

        # 3. Loss & Backprop
        loss = F.smooth_l1_loss(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()