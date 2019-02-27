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
from tqdm import tqdm

# Finds Image Gradiant in X and Y Directions
def getGradients(image):

	dx = np.zeros(image.shape, dtype=np.float64)
	dy = np.zeros(image.shape, dtype=np.float64)
	for i in range(image.shape[0]):
		for j in range(0, image.shape[1]-1):
			dx[i, j] = float(image[i, j+1]) - float(image[i, j])
		# dx[i, image.shape[1]-1] = -image[i, j]
	for j in range(image.shape[1]):
		for i in range(0, image.shape[0]-1):
			dy[i, j] = float(image[i+1, j]) - float(image[i, j])
		# dy[i, image.shape[0]-1] = -image[i, j]		

	return dx, dy
def Thresholding(image, thresh, kSize):
	img1 = np.zeros(image.shape, dtype=np.uint8)
	img2 = np.zeros(image.shape, dtype=np.uint8)
	img3 = np.zeros(image.shape, dtype=np.uint8)
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			for k in range(image.shape[2]):
				img1[i, j, k] = image[i, j, k]
				img2[i, j, k] = image[i, j, k]
				img3[i, j, k] = image[i, j, k]

	# result = np.load("q3_img/corner_results.npy")
	# rvalues = np.load("q3_img/rvalue_results.npy")
	RMatrix = np.load("q3_img/RMatrix.npy")
	# print(result.shape)
	# print(result)
	kCenterX = kSize//2
	kCenterY = kSize//2
	for i in tqdm(range(kCenterX, RMatrix.shape[0]-kCenterX)):
		for j in range(kCenterY, RMatrix.shape[1]-kCenterY):
			center = RMatrix[i, j]
			flag = 1
			for k in range(kSize):
				for l in range(kSize):
					row = i - kCenterX + k
					col = j - kCenterY + l
					if ( row >= 0 and row < RMatrix.shape[0] and col >= 0 and col < RMatrix.shape[1] ):
						if (row != i and col != j and RMatrix[row, col] < center ):
							flag = 0
							break
				if flag == 0:
					break
			if flag == 1:
				for thr in range(len(thresh)):
					ig = None
					if center > thresh[thr]:
						if thr == 0:
							ig = img1
						elif thr == 1:
							ig = img2
						else:
							ig = img3
						ig[i-1, j-1, 0] = 255
						ig[i-1, j-1, 1] = 0
						ig[i-1, j-1, 2] = 0
						ig[i, j-1, 0] = 255
						ig[i, j-1, 1] = 0
						ig[i, j-1, 2] = 0
						ig[i+1, j-1, 0] = 255
						ig[i+1, j-1, 1] = 0
						ig[i+1, j-1, 2] = 0

						ig[i-1, j, 0] = 255
						ig[i-1, j, 1] = 0
						ig[i-1, j, 2] = 0
						ig[i+1, j, 0] = 255
						ig[i+1, j, 1] = 0
						ig[i+1, j, 2] = 0

						ig[i-1, j+1, 0] = 255
						ig[i-1, j+1, 1] = 0
						ig[i-1, j+1, 2] = 0
						ig[i, j+1, 0] = 255
						ig[i, j+1, 1] = 0
						ig[i, j+1, 2] = 0
						ig[i+1, j+1, 0] = 255
						ig[i+1, j+1, 1] = 0
						ig[i+1, j+1, 2] = 0
						
				# if center > thresh[0]:
				# 	# img1[i, j, 0] = 255
				# 	# img1[i, j, 1] = 0
				# 	# img1[i, j, 2] = 0
					
				# if center > thresh[1]:
				# 	img2[i, j, 0] = 255
				# 	img2[i, j, 1] = 0
				# 	img2[i, j, 2] = 0
				# if center > thresh[2]:
				# 	img3[i, j, 0] = 255
				# 	img3[i, j, 1] = 0
				# 	img3[i, j, 2] = 0


	# for i in tqdm(range(len(result))):
	# 	x = int(result[i, 0])
	# 	y = int(result[i, 1])
	# 	R = float(rvalues[i])
	# 	if R >= thresh[0]:
			# img1[x, y, 0] = 255
			# img1[x, y, 1] = 0
			# img1[x, y, 2] = 0
	# 	if R >= thresh[1]:
	# 		img2[x, y, 0] = 255
	# 		img2[x, y, 1] = 0
	# 		img2[x, y, 2] = 0
	cv2.imwrite("q3_img/T1_HCD.png", img1)
	cv2.imwrite("q3_img/T2_HCD.png", img2)
	cv2.imwrite("q3_img/T3_HCD.png", img3)

def HarrisCornerDetection(image, kSize, sD):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	kernel = MyGaussianKernel(kSize, kSize, sD)
	# print(kernel)
	kCenterX = kSize//2
	kCenterY = kSize//2
	# xy_result = []
	# r_result = []

	RMatrix = np.zeros(image.shape, dtype=np.float64)

	for i in tqdm(range(kCenterX, image.shape[0]-kCenterX)):
		for j in range(kCenterY, image.shape[1]-kCenterY):
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
			Ixy = np.zeros((kSize, kSize), dtype=np.float64)
			for k in range(kSize):
				for l in range(kSize):
					Ixy[k, l] = Ix[k, l] * Iy[k, l]

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
			# print("M:", M)

			w = LA.eigvals(M)
			R = w[0]*w[1] - (0.05)*(w[0]+w[1])*(w[0]+w[1])
			RMatrix[i, j] = R
			# xy_result.append([i, j])
			# r_result.append(R)
			# print(Results)
	# np.save("q3_img/corner_results.npy", np.asarray(xy_result, dtype=np.int64))
	# np.save("q3_img/rvalue_results.npy", np.asarray(r_result, dtype=np.float64))
	np.save("q3_img/RMatrix.npy", RMatrix)
	print("----Harris Corner Detection Completed----")

			


# In[]       
Image = cv2.imread("q3_img/flower.jpg", 1)
# Image = cv2.imread("q3_img/flower.jpg", 1)
# HarrisCornerDetection(Image, 7, 1.5)
Thresholding(Image, [10, 100, 1000], 3)

