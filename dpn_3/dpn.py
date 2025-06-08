import torch
import torch.nn as nn
import torch.nn.functional as F

class DPN(nn.Module):
    def __init__(self, input_features, total_nodes, output_nodes, minimal=True):
        super().__init__()
        self.minimal = minimal
        self.output_nodes = output_nodes
        self.total_nodes = total_nodes
        if minimal:
            hidden_nodes = total_nodes - output_nodes
            self.layer1 = nn.Linear(input_features, hidden_nodes)
            self.layer2 = nn.Linear(input_features + hidden_nodes, output_nodes)

        else:
            self.layers = nn.ModuleList()
            for node in range(total_nodes):
                self.layers.append(nn.Linear(input_features + node, 1))

    def forward(self, x):
        if self.minimal:
            h1 = self.layer1(x)
            # Optional: apply nonlinearity, e.g. ReLU
            h1 = F.relu(h1)
            # Concatenate input and h1 along the feature dimension
            concat = torch.cat([x, h1], dim=1)
            out = self.layer2(concat)
            return out
        else:
            hidden_nodes = self.total_nodes - self.output_nodes
            for node in range(self.total_nodes):
                out = self.layers[node](x)
                if node < hidden_nodes:
                    out = F.relu(out)
                x = torch.cat([x, out], dim=1)

            return x[:, -self.output_nodes:]