# @Author: Kaustav Vats 
# @Roll-Number: 2016048 

# In[]
# Detect Circles using Hough Transform
# In[]
from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
from helper import GaussianFilter
import math
from tqdm import tqdm

# Hough Transform Function
class HoughTransform:

    def __init__(self, img, imageo):
        self.Orig = imageo
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                self.image[i, j] = img[i, j]
        self.RadiusRange = [10, min(self.image.shape[0]//2, self.image.shape[1]//2)]
        # self.RadiusRange = [5, ]
        print(self.RadiusRange)
        self.Accumulator = np.zeros((self.image.shape[0], self.image.shape[1], self.RadiusRange[1]-self.RadiusRange[0]), dtype=np.int64)
        print(self.Accumulator.shape)

    def run(self):
        for i in tqdm(range(self.image.shape[0])):
            for j in range(self.image.shape[1]):
                if self.image[i, j] == 255:
                    for r in range(self.RadiusRange[0], self.RadiusRange[1]):                
                        for theta in range(0, 360):
                            a = int(i - math.ceil(r * math.cos((theta * math.pi)/180)))
                            b = int(j - math.ceil(r * math.sin((theta * math.pi)/180)))
                            # print("a=", a, "\tb=", b, "\tr=", r-self.RadiusRange[0])
                            if a < 0 or b < 0 or a >= self.image.shape[0] or b >= self.image.shape[1]:
                                continue
                            self.Accumulator[a, b, r-self.RadiusRange[0]] += 1
            # print("i:", i)
        np.save("q1_img/hough.npy", self.Accumulator)
        # self.CheckAcc()

    def CheckAcc(self):
        self.Accumulator = np.load("q1_img/hough.npy")
        maxvalue = -1
        minvalue = math.inf
        for r in range(self.RadiusRange[0], self.RadiusRange[1]):
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    if self.Accumulator[i, j, r-self.RadiusRange[0]] > maxvalue:
                        maxvalue = self.Accumulator[i, j, r-self.RadiusRange[0]]
                    if self.Accumulator[i, j, r-self.RadiusRange[0]] < minvalue:
                        minvalue = self.Accumulator[i, j, r-self.RadiusRange[0]]
        x = -1
        y = -1
        print(maxvalue, minvalue)
        # large = 0
        for r in tqdm(range(self.RadiusRange[0], self.RadiusRange[1])):
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    if self.Accumulator[i, j, r-self.RadiusRange[0]] > int((6*math.pi*r)/8):
                        # large = self.Accumulator[i, j, r-self.RadiusRange[0]]
                        x = i
                        y = j
                        # print("self.Acc:", self.Accumulator[x, y, r-self.RadiusRange[0]])
                        self.Orig[x, y, 0] = 0
                        self.Orig[x, y, 1] = 0
                        self.Orig[x, y, 2] = 255
                        for theta in range(0, 360):
                            a = int(i + (math.ceil(r * math.cos((theta * math.pi)/180))))
                            b = int(j + (math.ceil(r * math.sin((theta * math.pi)/180))))
                            # print("a=", a, "\tb=", b, "\tr=", r-self.RadiusRange[0])
                            if a < 0 or b < 0 or a >= self.image.shape[0] or b >= self.image.shape[1]:
                                continue
                            self.Orig[a, b, 0] = 0
                            self.Orig[a, b, 1] = 0
                            self.Orig[a, b, 2] = 255
                            
                        # print("x:", x, "y:", y)
        cv2.imwrite("q1_img/hough_result.png", self.Orig)
        print("----Hough Transform Done----")
        

# Q1 ------------------------------------

# In[]
## Applyting Gaussian Filter and Canny Edge Detector
# In[]
# Image = cv2.imread('q1_img/custom.png') # 1, 0, -1
Image = cv2.imread('q1_img/Q1.jpeg') # 1, 0, -1
print("Image.shape", Image.shape)
Image2 = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
# Result1 = GaussianFilter(Image2, 7, 7, 1.5)
# cv2.imwrite("q1_img/Gaussian.png", Result1)
# np.save("q1_img/GResult.npy", Result1)
# In[]
Result1 = np.load("q1_img/GResult.npy")
edges = cv2.Canny(Result1, 60, 100)
# edges = cv2.Canny(Result1, 60, 100)
# kernel = np.ones((5, 5), np.uint8)
# edges = cv2.erode(edges, kernel, iterations=1) 
cv2.imwrite("q1_img/Canny.png", edges)
# cv2.imshow("Canny", edges)

# In[]
## Hough Transform
# In[]
ht = HoughTransform(edges, Image)
# In[]
ht.run()

# In[]
ht.CheckAcc()


# In[]
