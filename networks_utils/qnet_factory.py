from networks_utils.cnn import CNNNetwork
from networks_utils.nn import MLPNetwork

class QnetFactory:
    @staticmethod
    def create_qnet(input_shape, n_actions, config=None):
        """
        Détection automatique basée sur la forme de l'input.
        input_shape (tuple): (C, H, W) -> CNN, (N,) -> MLP
        """
        if len(input_shape) == 3:
            return CNNNetwork(input_shape, n_actions, config=config)
        else:
            return MLPNetwork(input_shape, n_actions, config=config)