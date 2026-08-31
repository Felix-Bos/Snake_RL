import torch
import torch.nn as nn

def get_activation(act_name):
    if act_name is None: return nn.ReLU()
    act_name = act_name.lower()
    if act_name == "relu": return nn.ReLU()
    if act_name == "tanh": return nn.Tanh()
    if act_name == "sigmoid": return nn.Sigmoid()
    if act_name == "leaky_relu": return nn.LeakyReLU()
    if act_name == "elu": return nn.ELU()
    return nn.ReLU()

class CNNNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, config=None):
        super(CNNNetwork, self).__init__()
        
        # Exemple de config par défaut robuste
        if config is None:
            config = {
                'conv_layers': [
                    {'out': 32, 'kernel': 8, 'stride': 4, 'activation': 'relu'},
                    {'out': 64, 'kernel': 4, 'stride': 2, 'activation': 'relu'},
                    {'out': 64, 'kernel': 3, 'stride': 1, 'activation': 'relu'},
                ],
                'fc_layers': [
                    {'size': 512, 'activation': 'relu'}
                ]
            }

        conv_layers_conf = config.get('conv_layers', [])
        fc_layers_conf = config.get('fc_layers', [])
        
        # Valeur par défaut globale si jamais oublié dans une couche spécifique
        default_act = config.get('activation', 'relu') 

        # --- 1. Construction Convolutions ---
        layers = []
        in_channels = input_shape[0]
        
        for layer_conf in conv_layers_conf:
            out_channels = layer_conf['out']
            kernel = layer_conf['kernel']
            stride = layer_conf['stride']
            padding = layer_conf.get('padding', 0)
            
            # On cherche l'activation spécifique à cette couche, sinon défaut global
            act_name = layer_conf.get('activation', default_act)
            
            layers.append(nn.Conv2d(in_channels, out_channels, kernel, stride, padding))
            layers.append(get_activation(act_name))
            in_channels = out_channels
            
        layers.append(nn.Flatten())
        self.conv = nn.Sequential(*layers)
        
        # --- 2. Calcul taille Flatten ---
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            n_flatten = self.conv(dummy).shape[1]

        # --- 3. Construction Fully Connected ---
        fc_layers = []
        last_dim = n_flatten
        
        for layer_conf in fc_layers_conf:
            # Gestion flexible : soit un entier (taille), soit un dict (taille + activation)
            if isinstance(layer_conf, int):
                hidden_size = layer_conf
                act_name = default_act
            else:
                hidden_size = layer_conf['size']
                act_name = layer_conf.get('activation', default_act)

            fc_layers.append(nn.Linear(last_dim, hidden_size))
            fc_layers.append(get_activation(act_name))
            last_dim = hidden_size
            
        fc_layers.append(nn.Linear(last_dim, n_actions))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.fc(self.conv(x))