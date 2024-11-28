import numpy as np
import scipy
import torch
import torch.nn as nn
from torchdiffeq import odeint as torchodeint
from tqdm import tqdm

class KANODE():
    def __init__(self,
                 model
                 ):

        self.model = model
        #self.ode = ode
        #self.odeInitialConditions = odeInitialConditions
        #self.odeVariables = odeVariables
        #self.integrationTime = integrationTime
        #self.trainingTime = trainingTime
        #self.trainingSamples = trainingSamples
        #self.epochs = epochs
        #self.plotFrequency = plotFrequency

        self.trainLossArray = np.array([])
        self.testLossArray = np.array([])


    def train(self,
              baseSolution,
              ode,
              odeInitialConditions,
              odeVariables,
              integrationTime,
              trainingTime,
              trainingSamples,
              epochs,
              plotFrequency,
              recordEval = True):
        
        #numTimeSamples = int(trainingSamples * (integrationTime/trainingTime))
        #numTimeSamples = int((trainingSamples/trainingTime) * integrationTime)
        #time = np.linspace(0, integrationTime, numTimeSamples)
        #baseSolution = scipy.integrate.odeint(ode, odeInitialConditions, time, args=(*odeVariables,))

        X0=torch.unsqueeze((torch.Tensor(np.transpose(odeInitialConditions))), 0)
        X0.requires_grad=True
        baseSolution=torch.Tensor(baseSolution)
        baseSolution.requires_grad=True
        trainData = baseSolution[:trainingSamples, :]
        time = torch.Tensor(time)
        trainTime = time[:trainingSamples, :]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3)

        modelWrapper = lambda x, t: self.model(x)

        for epoch in tqdm(range(epochs)):
            self.model.train()
            optimizer.zero_grad()

            prediction = torchodeint(modelWrapper, X0, trainTime)
            trainLoss = torch.mean(torch.square(prediction[:, 0, :]-trainData))
            trainLoss.retain_grad()
            trainLoss.backward()
            optimizer.step()
            self.trainLossArray.append(trainLoss.detach().cpu())

            if recordEval:
                self.testLossArray.append(self.test())



    def test(self, baseSolution):

        modelWrapper = lambda x, t: self.model(x)

        self.model.eval()
        prediction = torchodeint(modelWrapper, X0, trainTime)