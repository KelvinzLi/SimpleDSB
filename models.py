import torch
import torch.nn as nn
import torch.nn.functional as F
    
class MLP(nn.Module):
    def __init__(self, state_dim, t_dim, hidden_dim, num_layers, step_size = None):
        super().__init__()
        
        layers = []
        for ii in range(num_layers):
            input_dim = state_dim + t_dim if ii == 0 else hidden_dim
            output_dim = state_dim if ii == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(input_dim, output_dim))
        
        self.layers = nn.ModuleList(layers)
        self.step_size = step_size

    def forward(self, x, t_embeddings):
        og = x
        x = torch.cat([x, t_embeddings], dim = -1)
        
        for ii in range(len(self.layers)):
            x = self.layers[ii](x)
            if ii != len(self.layers) - 1:
                x = F.relu(x)
        
        if self.step_size is None:
            return x
        else:
            return og + self.step_size * x