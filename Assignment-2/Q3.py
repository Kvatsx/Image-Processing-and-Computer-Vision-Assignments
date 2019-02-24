# @Author: Kaustav Vats 
# @Roll-Number: 2016048 

# In[]
from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
from helper import MyGaussianKernel

# def getGradients(image, kernel):


def HarrisCornerDetection(image, kSize, sD, thresholds):
    kernel = MyGaussianKernel(kSize, kSize, sD)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

