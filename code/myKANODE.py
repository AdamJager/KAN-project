import numpy as np
import scipy
import torch
import torch.nn as nn
from torchdiffeq import odeint as torchodeint
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


class KANODE():
    def __init__(self,
                 model,
                 ode,
                 odeInitialState,
                 odeParameters,
                 integrationTime,
                 trainingTime,
                 stepSize
                 ):

        """
        A class that allows for convenient and modular use of the NNODE method on any neural network model with any (2d) ode
        
        Inputs:
            model: This should be a neural network object inheriting from torch.nn.module and have a defined forwards pass function.
            ode: This can be any function that takes in a vector of time points and a set of initial conditions. The function the model learns
            odeInitialState: Vector used as the initial conditions to calculate the given ode
            odeParameters: Vector used to input any parameters required for the ode function
            integrationTime: The amount of time in seconds that the ode should be calculated for
            trainingTime: The amount of time in seconds that the model is allowed to learn from
            stepSize: In seconds how large a single step in the base solution integrator is
        """



        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        #self.model.to(device)

        self.ode = ode
        self.odeParameters = odeParameters
        self.integrationTime = integrationTime
        self.trainingTime = trainingTime
        self.trainingSamples = int(trainingTime/stepSize)

        self.odeInitialState = torch.unsqueeze((torch.Tensor(np.transpose(odeInitialState))), 0)
        self.odeInitialState.requires_grad=True
        baseSolution = self.calculateBaseSolution()
        #normBaseSolution = self.standardiseData(np.transpose(baseSolution))
        #self.baseSolution = torch.Tensor(np.transpose(normBaseSolution))
        self.baseSolution = torch.Tensor(baseSolution)
        self.baseSolution.requires_grad=True

        self.trainLossArray = np.array([])
        self.testLossArray = np.array([])
        self.time = torch.tensor(self.calculateTimeArray())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3)

    def calculateBaseSolution(self):

        """ This function is used to calculate the solution to the provided ODE using a scipy integrator.
            This base solution provided by the scipy integrator will be the 'true' soution the model learns from
            and evaultaes against.
            
            Returns:
                numpy array: array of floats represnting the value of the ode at time t specified at the creation of the class"""

        time = self.calculateTimeArray()
        baseSolution = scipy.integrate.odeint(self.ode, self.odeInitialState.detach()[0], time, args=(*self.odeParameters,))
        return np.array(baseSolution)

    def calculateTimeArray(self):
        numTimeSamples = int((self.trainingSamples/self.trainingTime) * self.integrationTime)
        time = np.linspace(0, self.integrationTime, numTimeSamples)
        return time

    def standardiseData(self, data, lowerBound = -0.9, upperBound = 0.9): # not used xd
        normData = np.zeros(data.shape)
        for rowIndex, row in enumerate(data):
            normData[rowIndex, :] = (upperBound - lowerBound) * ((row - np.min(row)) / (np.max(row) - np.min(row))) + lowerBound
        return normData


    def train(self,
              epochs,
              recordEval = True,
              maxEpochs = 10000,
              saveCooldown = 200,
              evalThreshold = 10**-5,
              saveDirectory = os.getcwd(),
              saveName = "defaultName ",
              rtol = 1e-7,
              atol = 1e-9,
              solverMethod = "dopri5"):
        
        """
        The main training loop for the model. The loop records training loss and by default evaluation loss at each epoch. These are then appended to the testLossArray and trainLossArray. This allows
        for consecutive training batches without needing to exporting and piecing data together manually (convenient for plotting).

        Inputs:
            epochs: Integer denoting how many passes over the data the model gets. Can be set to -1 to run indefinitely or up to a stopping condition.
            recordEval: Boolean flag that determines wether the model records the evaluation loss (set to true by default).
            maxEpochs: termination condition if epochs is set to -1.
            saveCooldown: how many epochs should be taken between model checkpoints, models are only saved if a new best loss is reached.
                          Checkpoints are only created if the epochs parameter is set to -1, otherwise the model needs to be saved manually after training (using the "saveModel" function).
            evalThreshold: termination condition based on the evaluation metric. Training stops if the test score goes below threshold.
            saveDirectory: os path that determines where model checkpoints are stored, set to current working directory by default.
            saveName: string that is used as the name of the file containing saved models. Current Epoch is appended to the end.
            rtol: relative tollerance parameter for dynamic solvers
            atol: absolute tollerance parameter for dynamic solvers
            solverMethod: String specifiying which solver to use, set to "dopri5" by default (for backwards compatibility with my first scripts). CAUTION -- dopri5 is an adaptive step size
                          solver and will create an underflow error if the step size becomes too small (a common problem when working with stiff equations) to meet the rtol/atol requirements.
                          This can be fixed by adjusting the rtol/atol parameters or by switching to a different solver like rk4.
            
            
        """

        save = False
        if epochs == -1:
            epochs = maxEpochs
            save = True     
    
        trainData = self.baseSolution[:self.trainingSamples, :]
        trainTime = self.time[:self.trainingSamples]

        modelWrapper = lambda t, x: self.model(x) # using a lambda function because the torchodeint solver requires time and position as inputs to a function but the model only takes in position
                                                  # remember that we are using a NN to approximate an ode so it needs to have the same form for the numerical solver
        trainLossArray = np.zeros(epochs)
        if recordEval == True:
            testLossArray = np.zeros(epochs)

        bestLoss = 10000 # temp value to be updated in loop
        lastRecord = 0 # temp value showing when the last save was
       

        for epoch in tqdm(range(epochs)): # using tqdm to display progress bar at runtime
            self.model.train()
            self.optimizer.zero_grad()


            prediction = torchodeint(modelWrapper, self.odeInitialState, trainTime, rtol=rtol, atol=atol, method=solverMethod)


            trainLoss = torch.mean(torch.square(prediction[:, 0, :]-trainData))
            trainLoss.retain_grad()
            trainLoss.backward()
            self.optimizer.step()
            trainLossArray[epoch] = trainLoss.detach().cpu()

            if recordEval:
                testLoss, _ = self.test(solverMethod, rtol=rtol, atol=atol)
                testLossArray[epoch] = testLoss

            if save == False:
                continue

            if (testLoss < bestLoss) and (epoch > (lastRecord + saveCooldown)):
                self.saveModel(saveDirectory, saveName + f"{epoch}", epoch, testLoss)
                bestLoss = testLoss
                lastRecord = epoch

            if testLoss < evalThreshold:
                self.saveModel(saveDirectory, saveName + f"{epoch}", epoch, testLoss)
                break

        self.trainLossArray = np.append(self.trainLossArray, trainLossArray)
        self.testLossArray = np.append(self.testLossArray, testLossArray)

    def test(self, solverMethod="dopri5", rtol=1e-7, atol=1e-9):

        modelWrapper = lambda t, x: self.model(x)
        self.model.eval()
        prediction = torchodeint(modelWrapper, self.odeInitialState, self.time, rtol=rtol, atol=atol, method=solverMethod)
        loss = torch.mean(torch.square(prediction[self.trainingSamples:,0, :]-self.baseSolution[self.trainingSamples:, :])).detach().cpu()

        return loss, prediction


    def saveModel(self, modelDirectory, modelName, epoch, loss=None):
            
            """
            Saves the model state (all parameters specified in the model), optimizer state (requierd if trainig needs to be resumed),
            current loss (test loss if saved from within the train loop), and the current epoch.

            note: loss is set to 'None' by default, this might cause errors if manually saving a model. idk
            """

            if not os.path.isdir(modelDirectory):
                os.makedirs(modelDirectory)
            modelPath = os.path.join(modelDirectory, f"{modelName}.pt")
            torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        'epoch': epoch
            }, modelPath)

    def loadModel(self, modelDirectory, modelName, loadOptimizer = True, returnCheckPoint = False):

        """
        Loads a model from a previous save, this will replace the model and optimizer but doesn't clear any other parameters in the KANODE class object.
        This may lead to unintended behaviour if you train a model then load a different model and continue training as the training loop appends loss rather than overwriting.
        """

        if not os.path.exists(modelDirectory):
            print("This path does not exist")
            return

        modelPath = os.path.join(modelDirectory, f"{modelName}.pt")
        if not os.path.exists(modelPath):
            print("This model does not exist")

        checkpoint = torch.load(modelPath, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if loadOptimizer == True:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if returnCheckPoint == True:
            return checkpoint


    def legacyLoadModel(self, modelDirectory, modelName, loadOptimizer = True):

        """
        An old and strictly worse (possibly buggy) loading function that I made to keep access to models that were trained before I updated how everything was saved. Should NOT be used if not absolutely necessary.
        """

        if not os.path.exists(modelDirectory):
            print("This path does not exist")
            return

        modelPath = os.path.join(modelDirectory, f"{modelName}.pt")
        if not os.path.exists(modelPath):
            print("This model does not exist")

        self.model = torch.load(modelPath)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3)
        
        if loadOptimizer == True:
            optimizerPath = os.path.join(modelDirectory, f"{modelName}_optimizer.pt")
            self.optimizer.load_state_dict(torch.load(optimizerPath))


    # A whole bunch of setters that ensure relevant parameters are updated when changes are made. please use these to avoid unintended behaviour :) (unfortunately I don't think I tested all of them thoroughly so be careful :/)
    def setModel(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3)

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
        self.baseSolution = torch.Tensor(self.calculateBaseSolution())
        self.baseSolution.requires_grad=True

    def setTrainingTime(self, trainingTime):
        self.trainingTime = trainingTime
        self.time = torch.tensor(self.calculateTimeArray())
        self.baseSolution = torch.Tensor(self.calculateBaseSolution())
        self.baseSolution.requires_grad=True

    # forgot to add a setter for stepsize when I updated the system to work using stepsize rather than number of steps. This should be changed if needed for consitency -- sorry I'm lazy ;_;
    def setTrainingSamples(self, trainingSamples):
        self.trainingSamples = trainingSamples
        self.time = torch.tensor(self.calculateTimeArray())
        self.baseSolution = torch.Tensor(self.calculateBaseSolution())
        self.baseSolution.requires_grad=True


    # plotting functions
    def plotODE(self, title, solution, startTime=0):
        plt.figure()
        plt.title(title)
        plt.plot(self.time, self.baseSolution[:, 0].detach(), color='g')
        plt.plot(self.time, self.baseSolution[:, 1].detach(), color='b')
        plt.plot(self.time, solution[:, 0].detach(), linestyle='dashed', color='tab:olive')
        plt.plot(self.time, solution[:, 1].detach(), linestyle='dashed', color='tab:cyan')

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
        plt.plot(self.baseSolution[:, 0].detach(), self.baseSolution[:, 1].detach(), label="Base Solution")
        plt.plot(solution[:, 0].detach(), solution[:, 1].detach(), label="Predicted Solution")
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()