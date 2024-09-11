import torch.nn as nn
import torch.nn.functional as F


class JosaPcc(nn.Module):
    def __init__(self, in_feature=8, neurons=8, out_features=2, layer_num=5):
        """
        The PCC model, i.e., Simple MLP net based on 5 hidden layers, with 2 linear layer,
         i.e., the first_layer and last_layer.
        """
        super(JosaPcc, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_feature if i == 0 else neurons, 
                      out_features if i == layer_num else neurons)
            for i in range(layer_num + 1)
        ])

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x
