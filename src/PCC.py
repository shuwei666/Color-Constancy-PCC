import torch.nn as nn
from torch.nn import functional as F


class JosaPcc(nn.Module):
    def __init__(self, in_feature=8, neurons=8, out_features=2, layer_num=5):
        """
        The PCC model, i.e., Simple MLP net based on 5 hidden layers, with 2 linear layer,
         i.e., the first_layer and last_layer.
        """
        super(JosaPcc, self).__init__()
        self.first_layer = nn.Linear(in_feature, neurons)
        self.hidden_layer = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(layer_num)])
        self.last_layer = nn.Linear(neurons, out_features)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.first_layer(x)

        for layer in self.hidden_layer:
            x = F.relu(layer(x))

        out = self.last_layer(x)

        return out
