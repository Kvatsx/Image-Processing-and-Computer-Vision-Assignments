# @Author: Kaustav Vats 
# @Roll-Number: 2016048 

# In[]
from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
from helper import MyGaussianKernel
from numpy import linalg as LA
from math import log10

# Finds Image Gradiant in X and Y Directions
def getGradients(image):

	dx = np.zeros(image.shape, dtype=np.float64)
	dy = np.zeros(image.shape, dtype=np.float64)
	for i in range(image.shape[0]):
		for j in range(0, image.shape[1]-1):
			dx[i, j] = float(image[i, j+1]) - float(image[i, j])
		dx[i, image.shape[1]-1] = -image[i, j]
	for j in range(image.shape[1]):
		for i in range(0, image.shape[0]-1):
			dy[i, j] = float(image[i+1, j]) - float(image[i, j])
		dy[i, image.shape[0]-1] = -image[i, j]		

	return dx, dy


def HarrisCornerDetection(image, kSize, sD, thresholds):
	img1 = np.zeros(image.shape, dtype=np.uint8)
	img2 = np.zeros(image.shape, dtype=np.uint8)
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			for k in range(image.shape[2]):
				img1[i, j, k] = image[i, j, k]
				img2[i, j, k] = image[i, j, k]

	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	kernel = MyGaussianKernel(kSize, kSize, sD)
	kCenterX = kSize//2
	kCenterY = kSize//2
	# dx, dy = getGradients(image)
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			# Ix = np.zeros((kSize, kSize), dtype=np.float64)
			# Iy = np.zeros((kSize, kSize), dtype=np.float64)
			SubImage = np.zeros((kSize, kSize), dtype=np.uint8)

			for k in range(kSize):
				for l in range(kSize):
					row = i - kCenterX + k
					col = j - kCenterY + l
					if ( row >= 0 and row < image.shape[0] and col >= 0 and col < image.shape[1] ):
						SubImage[k, l] = image[row, col]
			Ix, Iy = getGradients(SubImage)
			Ix2 = np.square(Ix)
			Iy2 = np.square(Iy)
			Ixy = np.matmul(Ix, Iy)

			for k in range(kSize):
				for l in range(kSize):
					Ix2[k, l] = Ix2[k, l] * kernel[k, l]
					Iy2[k, l] = Iy2[k, l] * kernel[k, l]
					Ixy[k, l] = Ixy[k, l] * kernel[k, l]
			M = np.zeros((2, 2), dtype=np.float64)
			M[0, 0] = np.sum(Ix2)
			M[1, 1] = np.sum(Iy2)
			M[1, 0] = np.sum(Ixy)
			M[0, 1] = np.sum(Ixy)
			# print(np.sum(Ix2))
			w = LA.eigvals(M)
			# print(w)
			# print(w.shape)
			R = w[0]*w[1] - (0.05)*(w[0]+w[1])*(w[0]+w[1])
			# print("R:", R)
			# R = log10(R)
			if R >= thresholds[0]:
				img1[i, j, 0] = 255
				img1[i, j, 1] = 0
				img1[i, j, 2] = 0
			if R >= thresholds[1]:
				img2[i, j, 0] = 255
				img2[i, j, 1] = 0
				img2[i, j, 2] = 0
	cv2.imwrite("q3_img/T1_HCD.png", img1)
	cv2.imwrite("q3_img/T2_HCD.png", img2)
	print("----Harris Corner Detection Completed----")

			


# In[]       
Image = cv2.imread("q3_img/chess.png", 1)
HarrisCornerDetection(Image, 7, 1.5, [1000000, 10000000])

