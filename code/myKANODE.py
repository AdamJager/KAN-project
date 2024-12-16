import numpy as np
import scipy
import torch
import torch.nn as nn
from torchdiffeq import odeint as torchodeint
from tqdm import tqdm
import matplotlib.pyplot as plt


class KANODE():
    def __init__(self,
                 model,
                 ode,
                 odeInitialState,
                 odeParameters,
                 integrationTime,
                 trainingTime,
                 trainingSamples
                 ):

        self.model = model
        self.ode = ode
        self.odeParameters = odeParameters
        self.integrationTime = integrationTime
        self.trainingTime = trainingTime
        self.trainingSamples = trainingSamples

        self.odeInitialState = torch.unsqueeze((torch.Tensor(np.transpose(odeInitialState))), 0)
        self.odeInitialState.requires_grad=True
        baseSolution = self.calculateBaseSolution()
        self.baseSolution = torch.Tensor(baseSolution)
        self.baseSolution.requires_grad=True

        self.trainLossArray = np.array([])
        self.testLossArray = np.array([])
        self.time = torch.tensor(self.calculateTimeArray())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3)

    def calculateBaseSolution(self):
        time = self.calculateTimeArray()
        baseSolution = scipy.integrate.odeint(self.ode, self.odeInitialState.detach()[0], time, args=(*self.odeParameters,))       
        return baseSolution

    def calculateTimeArray(self):
        numTimeSamples = int((self.trainingSamples/self.trainingTime) * self.integrationTime)
        time = np.linspace(0, self.integrationTime, numTimeSamples)
        return time

    def train(self,
              epochs,
              #plotFrequency,
              recordEval = True):
        
        #numTimeSamples = int(trainingSamples * (integrationTime/trainingTime))
    
        trainData = self.baseSolution[:self.trainingSamples, :]
        trainTime = self.time[:self.trainingSamples]

        #optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3)

        modelWrapper = lambda t, x: self.model(x)

        trainLossArray = np.zeros(epochs)
        if recordEval == True:
            testLossArray = np.zeros(epochs)

        for epoch in tqdm(range(epochs)):
            self.model.train()
            self.optimizer.zero_grad()

            prediction = torchodeint(modelWrapper, self.odeInitialState, trainTime)
            trainLoss = torch.mean(torch.square(prediction[:, 0, :]-trainData))
            trainLoss.retain_grad()
            trainLoss.backward()
            self.optimizer.step()
            trainLossArray[epoch] = trainLoss.detach().cpu()

            if recordEval:
                testLoss, _ = self.test()
                testLossArray[epoch] = testLoss

            #if epoch % plotFrequency == 0:
                #self.plot()

        self.trainLossArray = np.append(self.trainLossArray, trainLossArray)
        self.testLossArray = np.append(self.testLossArray, testLossArray)

    def test(self):

        modelWrapper = lambda t, x: self.model(x)
        self.model.eval()
        prediction = torchodeint(modelWrapper, self.odeInitialState, self.time)
        loss = torch.mean(torch.square(prediction[self.trainingSamples:,0, :]-self.baseSolution[self.trainingSamples:, :])).detach().cpu()

        return loss, prediction
    
    def setModel(self, model):
        self.model = model

    def setODE(self, ode):
        self.ode = ode
        baseSolution = self.calculateBaseSolution()
        self.baseSolution = torch.Tensor(baseSolution)
        
    def setOdeIC(self, odeInitialState):
        self.odeInitialState = torch.unsqueeze((torch.Tensor(np.transpose(odeInitialState))), 0)
        self.odeInitialState.requires_grad=True
        baseSolution = self.calculateBaseSolution()
        self.baseSolution = torch.Tensor(baseSolution)
        
    def setodeParameters(self, odeParameters):
        self.odeParameters = odeParameters
        baseSolution = self.calculateBaseSolution()
        self.baseSolution = torch.Tensor(baseSolution)

    def setIntegrationTime(self, integrationTime):
        self.integrationTime = integrationTime
        self.time = torch.tensor(self.calculateTimeArray())
        self.baseSolution = torch.Tensor(self.calculateBaseSolution)
        self.baseSolution.requires_grad=True

    def setTrainingTime(self, trainingTime):
        self.trainingTime = trainingTime
        self.time = torch.tensor(self.calculateTimeArray())
        self.baseSolution = torch.Tensor(self.calculateBaseSolution)
        self.baseSolution.requires_grad=True

    def setTrainingSamples(self, trainingSamples):
        self.trainingSamples = trainingSamples
        self.time = torch.tensor(self.calculateTimeArray())
        self.baseSolution = torch.Tensor(self.calculateBaseSolution)
        self.baseSolution.requires_grad=True

    def plotODE(self, title, solution):
        plt.figure()
        plt.title(title)
        plt.plot(self.time, self.baseSolution[:, 0].detach(), color='g')
        plt.plot(self.time, self.baseSolution[:, 1].detach(), color='b')
        plt.plot(self.time, solution[:, 0].detach(), linestyle='dashed', color='g')
        plt.plot(self.time, solution[:, 1].detach(), linestyle='dashed', color='b')

        plt.legend(['x_data', 'y_data', 'x_KAN-ODE', 'y_KAN-ODE'])
        plt.ylabel('concentration')
        plt.xlabel('time')
        plt.vlines(self.trainingTime, 0, 5)

        plt.show()

    def plotLoss(self, title):
        plt.figure()
        plt.title(title)
        plt.semilogy(torch.Tensor(self.trainLossArray), label='train')
        plt.semilogy(torch.Tensor(self.testLossArray), label='test')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
 
        plt.show()

    def plotPhaseSpace(self, title, solution):
        plt.figure()
        plt.title(title)
        plt.plot(self.baseSolution[:, 0].detach(), self.baseSolution[:, 1].detach())
        plt.plot(solution[:, 0].detach(), solution[:, 1].detach())
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()