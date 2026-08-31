import torch
import torch.nn as nn

def get_activation(act_name):
    act_name = act_name.lower()
    if act_name == "relu": return nn.ReLU()
    if act_name == "tanh": return nn.Tanh()
    if act_name == "sigmoid": return nn.Sigmoid()
    if act_name == "leaky_relu": return nn.LeakyReLU()
    if act_name == "elu": return nn.ELU()
    return nn.ReLU()


class MLPNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, config=None):
        super(MLPNetwork, self).__init__()
        
        if config is None:
            config = {
                'hidden_layers': [256, 256],
                'activations': ['relu', 'relu']
            }
            
        input_dim = input_shape[0]
        layers = []
        
        hidden_sizes = config.get('hidden_layers', [256, 256])
        activations = config.get('activations', ['relu'] * len(hidden_sizes))
        
        # Si une seule activation est donnée pour tout le réseau
        if isinstance(activations, str):
            activations = [activations] * len(hidden_sizes)

        last_dim = input_dim
        
        for hidden_size, act_name in zip(hidden_sizes, activations):
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(get_activation(act_name))
            last_dim = hidden_size
            
        layers.append(nn.Linear(last_dim, n_actions))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)