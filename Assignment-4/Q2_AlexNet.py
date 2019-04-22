'''
 @Author: Kaustav Vats 
 @Roll-Number: 2016048 
 @Date: 2019-04-22 23:28:19 
'''

import numpy as np
import cv2
from utils import ReadData
import torch
import torch.nn as nn
import pickle
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

PATH = "./Data/"
IMAGE_SIZE = 214
MEAN = [0.485, 0.456, 0.406]
STD = [0.299, 0.244, 0.225]

'''---------------------Reading dataset----------------------''' 

X_Train, Y_Train, X_Test, Y_Test = ReadData()

'''---------------------Pre-Processing-----------------------'''

trans = transforms.Compose([
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(.3, .3, .3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


'''---------------------Features Extraction------------------'''

AlexNet = models.alexnet(pretrained=True)


'''---------------------Testing------------------------------'''


