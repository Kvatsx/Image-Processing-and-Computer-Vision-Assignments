# @Author: Kaustav Vats 
# @Roll-Number: 2016048 

# In[]
from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
from helper import GaussianFilter

def Thresholding(image):
    Result = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] < 128:
                Result[i, j] = 255
            else:
                Result[i, j] = 0
    print(Result)
    return Result


def LaplacianEdgeDetection(image, ksize):
    print("```Laplacian Edge Detection```")
    kernel = np.zeros((ksize, ksize), dtype=np.int)
    # if ksize == 3:
    #     kernel[0, 1] = -1
    #     kernel[2, 1] = -1
    #     kernel[1, 0] = -1
    #     kernel[1, 2] = -1
    #     kernel[1, 1] = 4


    kernel.fill(-1)
    if ksize == 3:
        kernel[ksize//2, ksize//2] = 8
    else:
        kernel[ksize//2, ksize//2] = 24
    print(kernel)

    Result = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    kCenterX = ksize//2
    kCenterY = ksize//2

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            num = 0.0
            for k in range(3):
                for l in range(3):
                    row = i - kCenterX + k
                    col = j - kCenterY + l
                    # print(row, col)
                    if ( row >= 0 and row < image.shape[0] and col >= 0 and col < image.shape[1] ):
                        num += float(float(image[row, col])*kernel[k, l])
            if num < 0:
                num = 0
            Result[i, j] = num
    # Result = Thresholding(Result)
    return Result


# Q1 ------------------------------------
# In[]

Image = cv2.imread('Q1.jpeg', 0) # 1, 0, -1
# cv2.imshow("Image", Image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print("Image.shape", Image.shape)
Result1 = GaussianFilter(Image, 5, 5, 1.5)
cv2.imwrite("Gaussian.png", Result1)

# In[]
edges = cv2.Canny(Result1, 60, 100)
cv2.imwrite("Canny.png", edges)
laplacian_edges = cv2.Laplacian(Result1, cv2.CV_16S, ksize=5, scale=0.5, delta=0, borderType=cv2.BORDER_DEFAULT)
cv2.imwrite("Laplacian_edges.png", laplacian_edges)

# In[]
custom_laplacian = LaplacianEdgeDetection(Result1, 5)
cv2.imwrite("customLap.png", custom_laplacian)

# In[]

custom_laplacian = Thresholding(custom_laplacian)
# print("CustomLap")
# print(custom_laplacian)
cv2.imwrite("Custom_laplacian.png", custom_laplacian)

# In[]


