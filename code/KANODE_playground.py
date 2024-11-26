# an attempt at writing better code, using cuda, and usinsg fasterkan 

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from torchdiffeq import odeint as torchodeint
from torchdiffeq import odeint_adjoint as torchodeint_adjoint
from tqdm import tqdm
import os
import gc
import torch.nn as nn
import sys
from kan import KAN as efficientkan #from efficient kan
import argparse

def pred_prey_deriv(X, t, alpha, beta, delta, gamma):
    x=X[0]
    y=X[1]
    dxdt = alpha*x-beta*x*y
    dydt = delta*x*y-gamma*y
    dXdt=[dxdt, dydt]
    return dXdt

def duffing_deriv(X, t, alpha, beta, delta, gamma, omega):
    x = X[0]
    y = X[1]
    y_dot = gamma*np.cos(omega*t) - beta*x**3 - alpha*x - delta*y
    return [y, y_dot]

#class KAN_Model(efficientkan):
#    def __init__(self, layers_hidden, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1, scale_spline=1, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=...):
#        super().__init__(layers_hidden, grid_size, spline_order, scale_noise, scale_base, scale_spline, base_activation, grid_eps, grid_range)

def create_parser():
    parser = argparse.ArgumentParser('KAN ode')            
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--test_time', type=float, default=3.5)
    parser.add_argument('--total_time', type=float, default=20)
    parser.add_argument('--time_step', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=2e-3)

    return parser

def get_function_data(func, sample_size, data_range=[0,1]):
    """
    function to get a random sample of an input func 
    func can be single or multi variate (sample_size variable must refelct the dimensions of func as [num_inputs, num_samples])
    

    Inputs:
   
    func: function that the user wants to sample
    sample_size: 1d array of size 2 refelcting number of inputs to func, and how many samples the user wants [num_inputs, num_samples]
    data_range: range of the input samples to the function, default value [0, 1)
    """
    samples = data_range[1] * np.random.sample(sample_size) - data_range[0]
    function_data = func(*samples)
    return function_data

def plotting():
    #create a plotting function

    return

def plot_phase_space(x, y):
    plt.figure()
    




if __name__ == 'main':

    parser = create_parser()  
    args = parser.parse_args()

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    model = efficientkan([2, 10, 2], 10)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1 + args.epochs):
        optimiser.zero_grad()
        #get data, maybe in loop idk
        #pedict
        #calc loss
        #back prop
        optimiser.step()

        #plotting logic perhaps
