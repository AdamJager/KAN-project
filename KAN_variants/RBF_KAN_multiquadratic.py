import torch
import torch.nn as nn
import torch.nn.init as init

class RBFKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_centers, alpha=1.0):
        super(RBFKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.alpha = alpha

        self.centers = nn.Parameter(torch.empty(num_centers, input_dim))
        init.xavier_uniform_(self.centers)

        self.weights = nn.Parameter(torch.empty(num_centers, output_dim))
        init.xavier_uniform_(self.weights)

    def multiquadratic_rbf(self, distances):
        return (1 + (self.alpha * distances) ** 2) ** 0.5

    def forward(self, x):
        distances = torch.cdist(x, self.centers)
        basis_values = self.multiquadratic_rbf(distances)
        output = torch.sum(basis_values.unsqueeze(2) * self.weights.unsqueeze(0), dim=1)
        return output

class RBFKAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_centers):
        super(RBFKAN, self).__init__()
        self.rbf_kan_layer = RBFKANLayer(input_dim, hidden_dim, num_centers)
        self.output_weights = nn.Parameter(torch.empty(hidden_dim, output_dim))
        init.xavier_uniform_(self.output_weights)

    def forward(self, x):
        x = self.rbf_kan_layer(x)
        x = torch.relu(x)
        x = torch.matmul(x, self.output_weights)
        return x