'''
 @Author: Kaustav Vats 
 @Roll-Number: 2016048 
 @Date: 2019-04-22 23:28:19 
 @Ref:- [
    https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/
    https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
    
    ]
'''

import numpy as np
import cv2
from utils import ReadData, ReadLabels, FilterData
import pickle
# import time
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim

PATH = "./Data/Q2-output/"

'''---------------------Reading dataset----------------------''' 

X_Train, Y_Train, X_Test, Y_Test = ReadData()
Labels = ReadLabels(["automobile", "cat", "dog", "truck"])
print("Labels Number: {}".format(Labels))

X_Train, Y_Train = FilterData(X_Train, Y_Train, Labels)
X_Test, Y_Test = FilterData(X_Test, Y_Test, Labels)

print("Train Data shape: {}".format(X_Train.shape))
print("Test Data shape: {}".format(X_Test.shape))

'''-------------------Convolution Neural Network-------------'''

class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()

        self.Conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.Conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.Conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.Linear1 = nn.Linear(64 * 32 * 32, 500)
        self.Linear1 = nn.Linear(64 * 32 * 32, 250)
        return

    def Forward(self, x):
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = F.relu(self.Conv3(x))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        return x

    def createLossAndOptimizer(self, net, learning_rate=0.001):
        loss = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        return(loss, optimizer)
