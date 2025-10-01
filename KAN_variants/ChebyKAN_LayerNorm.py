import torch
import torch.nn as nn
import os

# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y


    def setCoeffs(self, coeffs):
        self.cheby_coeffs = coeffs

class ChebyKAN(nn.Module):
    def __init__(self, layers, degree):
        super(ChebyKAN, self).__init__()


        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(ChebyKANLayer(layers[i], layers[i+1], degree))

        self.normLayers = nn.ModuleList()
        for i in range(len(layers) - 2):
            self.normLayers.append(nn.LayerNorm(layers[i+1]))


    def forward(self, x):
        for layer, normLayer in zip(self.layers, self.normLayers):
            x = layer(x)
            x = normLayer(x)
        x = self.layers[-1](x)
        return x
    