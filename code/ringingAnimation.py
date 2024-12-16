import numpy as np
import scipy
import torch
import torch.nn as nn
from torchdiffeq import odeint as torchodeint
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
adjacent_folder_path = os.path.join(current_dir, '..\\KAN_variants')
sys.path.append(adjacent_folder_path)
from ChebyKANLayer import ChebyKAN

module_path = os.path.dirname(os.path.join('..'))
sys.path.append(module_path+"\\code")
from myKANODE import KANODE

model = ChebyKAN([2, 10, 2], 4)

def pred_prey_deriv(X, t, alpha, beta, delta, gamma):
    x=X[0]
    y=X[1]
    dxdt = alpha*x-beta*x*y
    dydt = delta*x*y-gamma*y
    dXdt=[dxdt, dydt]
    return dXdt


params = [1.5, 1, 2, 1]
X0 = [0.2778, 1.5000] # dummy X0 for initialisation 

kanODE = KANODE(model, pred_prey_deriv, X0, params, 20, 6, 1000)



fig, ax = plt.subplots()

def update(frame):
    kanODE.train(1)
    _, solution = kanODE.test()
    solution = solution[:, 0, :]
    ax.clear()
    fig.legend(['x_data', 'y_data', 'x_KAN-ODE', 'y_KAN-ODE'])
    plt.ylabel('concentration')
    plt.xlabel('time')
    ax.set(xlim=[0, 20], ylim=[0, 5])
    plt.vlines(kanODE.trainingTime, 0, 5)
    ax.set_title(f"solution at epoch: {frame}")
    ax.plot(kanODE.time, kanODE.baseSolution[:, 0].detach(), color='g')
    ax.plot(kanODE.time, kanODE.baseSolution[:, 1].detach(), color='b')
    kanx = ax.plot(kanODE.time, solution[:, 0].detach(), linestyle='dashed', color='g')
    kany = ax.plot(kanODE.time, solution[:, 1].detach(), linestyle='dashed', color='b')

    return (kanx, kany)

ani = animation.FuncAnimation(fig=fig, func=update, frames=1000, interval=10)
ani.save(filename="ringinAnimation.gif", writer="pillow")

kanODE.plotLoss("animation gif loss  ")