# Kaustav Vats (2016048)

import numpy as np
import cv2
from tqdm import tqdm
from helper import *
from queue import Queue

# Functions For Skin Based Thresholding --------------------------------------
# Part 1 ------------------------------------------------------------------------------
def RGB_Threshold(bgr):
    b = float(bgr[0])
    g = float(bgr[1])
    r = float(bgr[2])

    E1 = r > 95 and g > 40 and b > 20 and ((max(r, max(g, b)) - min(r, min(g, b))) > 15) and (abs(r-g) > 15) and r > g and r > b
    E2 = r > 220 and g > 210 and b > 170 and abs(r-g) <= 15 and r > b and g > b

    return E1 or E2

def YCrCb_Threshold(yCrCb):
    y = float(yCrCb[0])
    Cr = float(yCrCb[1])
    Cb = float(yCrCb[2])

    E1 = Cr <= 1.5862*Cb+20
    E2 = Cr >= 0.3448*Cb+76.2069
    E3 = Cr >= -4.5652*Cb+234.5652
    E4 = Cr <= -1.15*Cb+301.75
    E5 = Cr <= -2.2857*Cb+432.85

    return E1 and E2 and E3 and E4 and E5

def HSV_Threshold(hsv):
    return hsv[0] < 25 or hsv[0] > 230

def SkinSegmentation(image, filename="./Q3-output/result.png"):
    result = np.copy(image)
    yCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    hsv = cv2.normalize(hsv, None, 0.0, 255.0, cv2.NORM_MINMAX, cv2.CV_32FC3)

    for i in tqdm(range(image.shape[0])):
        for j in range(image.shape[1]):
            a = RGB_Threshold(image[i, j])
            b = YCrCb_Threshold(yCrCb[i, j])
            c = HSV_Threshold(hsv[i, j])
            if (not (a and b and c)):
                result[i, j, 0] = 0
                result[i, j, 1] = 0
                result[i, j, 2] = 0

    cv2.imwrite(filename, result)

# Algorithm 2

def Threshold(bgra, hsv, yCrCb):
    b = float(bgra[0])
    g = float(bgra[1])
    r = float(bgra[2])
    a = float(bgra[3])
    y = float(yCrCb[0])
    Cr = float(yCrCb[1])
    Cb = float(yCrCb[2])

    E1 = hsv[0] < 50.0 and  0.23 <= hsv[1] <= 0.68 and r > 95 and g > 40 and b > 20 and r > g and r > b and abs(r-g) > 15 and a > 15
    E2 = r > 95 and g > 40 and b > 20 and r > g and r > b and abs(r-g) > 15 and a > 15 
    E2 = E2 and Cr > 135 and Cb > 85 and y > 80 and Cr <= (1.5862*Cb) + 20
    E2 = E2 and Cr >= (0.3448*Cb) + 76.2069
    E2 = E2 and Cr >= (-4.5652*Cb) + 234.5652
    E2 = E2 and Cr <= (-1.15*Cb) + 301.75
    E2 = E2 and Cr <= (-2.2857*Cb) + 432.85
    return E1 or E2

# https://arxiv.org/ftp/arxiv/papers/1708/1708.02694.pdf
def SkinAlgo2(image, filename="./Q3-output/result.png"):
    result = np.copy(image)
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    hsv = cv2.normalize(hsv, None, 0.0, 255.0, cv2.NORM_MINMAX, cv2.CV_32FC3)
    yCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    print(yCrCb.shape)

    for i in tqdm(range(image.shape[0])):
        for j in range(image.shape[1]):
            if (not Threshold(bgra[i, j], hsv[i, j], yCrCb[i, j])):
                result[i, j, 0] = 0
                result[i, j, 1] = 0
                result[i, j, 2] = 0

    cv2.imwrite(filename, result)

# Part3 ------------------------------------------------------------------------------
class SeedPointSegmentation:
    Image = None
    Visited = None
    Result = None
    SeedBGR = None
    Tolerance = None
    Average = None
    Count = None

    def __init__(self, image, seedPoint, tolerance):
        self.Image = image
        self.Tolerance = int(tolerance * 255 / 100)
        self.Visited = np.zeros((image.shape[0], image.shape[1]), dtype=np.int)
        self.Result = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        self.SeedBGR = [image[seedPoint[0], seedPoint[1], 0], image[seedPoint[0], seedPoint[1], 1], image[seedPoint[0], seedPoint[1], 2]]
        self.Average = [float(image[seedPoint[0], seedPoint[1], 0]), float(image[seedPoint[0], seedPoint[1], 1]), float(image[seedPoint[0], seedPoint[1], 2])]
        self.Count = 1

    def checkPoint(self, point):
        if point[0] > 0 and point[0] < self.Image.shape[0] and point[1] > 0 and point[1] < self.Image.shape[1]:
            if self.Visited[point[0], point[1]] == 1:
                return 0
            return 1
        return 0
    
    def updatePixel(self, point):
        B = abs(float(self.Image[point[0], point[1], 0]) - float(self.Average[0]/self.Count))
        G = abs(float(self.Image[point[0], point[1], 1]) - float(self.Average[1]/self.Count))
        R = abs(float(self.Image[point[0], point[1], 2]) - float(self.Average[2]/self.Count))
        
        if B < self.Tolerance and G < self.Tolerance and R < self.Tolerance:
            self.Result[point[0], point[1], 0] = self.Image[point[0], point[1], 0]
            self.Result[point[0], point[1], 1] = self.Image[point[0], point[1], 1]
            self.Result[point[0], point[1], 2] = self.Image[point[0], point[1], 2]
            self.Average[0] += float(self.Image[point[0], point[1], 0])
            self.Average[1] += float(self.Image[point[0], point[1], 1])
            self.Average[2] += float(self.Image[point[0], point[1], 2])
            self.Count += 1
            return 1
        return 0
    
    def ShowResults(self, filename):
        cv2.imwrite(filename, self.Result)

    def run(self, seedPoint):
        Q = Queue()
        Q.put([seedPoint[0], seedPoint[1]])
        self.Visited[seedPoint[0], seedPoint[1]] = 1

        while( Q.qsize() > 0 ):
            # print(Q.qsize())
            CurrentPoint = Q.get()
            Ret = self.updatePixel(CurrentPoint)
            if Ret == 0:
                continue

            p1 = [CurrentPoint[0]+1, CurrentPoint[1]]
            p2 = [CurrentPoint[0]-1, CurrentPoint[1]]
            p3 = [CurrentPoint[0], CurrentPoint[1]-1]
            p4 = [CurrentPoint[0], CurrentPoint[1]+1]

            if self.checkPoint(p1):
                Q.put(p1)
                self.Visited[p1[0], p1[1]] = 1
            if self.checkPoint(p2):
                Q.put(p2)
                self.Visited[p2[0], p2[1]]= 1
            if self.checkPoint(p3):
                Q.put(p3)
                self.Visited[p3[0], p3[1]]= 1
            if self.checkPoint(p4):
                Q.put(p4)
                self.Visited[p4[0], p4[1]] = 1

        

# Image Reading ----------------------------------------------------------------------------
ImageList = ["face1.jpg", "face2.jpg", "face3.jpg", "face4.jpg", "colors.jpg"]

Image = cv2.imread("./Q3-faces/" + ImageList[2])
print(Image.shape)
# SkinSegmentation(Image, "./Q3-output/" + ImageList[2])

SkinAlgo2(Image, "./Q3-output/x" + ImageList[2])

# Seed based segmentation ---------------------------------
# seedPoint = [600, 950]
# seedPoint = [250, 250]
# seedPoint = [200, 350]
# seedPoint = [120, 150]
# Seedps = SeedPointSegmentation(Image, seedPoint, 30)
# Seedps.run(seedPoint)
# Seedps.ShowResults("./Q3-output/p2_" + ImageList[3])
