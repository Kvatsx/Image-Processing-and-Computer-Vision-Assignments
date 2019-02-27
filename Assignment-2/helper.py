# @Author: Kaustav Vats 
# @Roll-Number: 2016048 

import numpy as np
import math
import matplotlib.pyplot as plt

def ShowResults(original, result, title):
    plt.subplot(121),plt.imshow(original),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(result),plt.title(title)
    plt.xticks([]), plt.yticks([])

def DownsampleMyImage(image):
    result = np.zeros((image.shape[0]//2, image.shape[1]//2), dtype=np.uint8)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            x = i*2
            y = j*2
            num = image[x, y]
            result[i, j] = num
    return result

def GaussianValue(x, y, var=1.0):
    s = 2.0*var*var
    val = math.exp(-((x*x+y*y) /s ))
    den = math.pi * s
    return val/den

def MyGaussianKernel(kSizeX, kSizeY, var):
    kernel = np.zeros((kSizeX, kSizeY), dtype=np.float64)
    kCenterX = kSizeX//2
    kCenterY = kSizeY//2
    total = 0.0
    for i in range(kSizeX):
        for j in range(kSizeY):
            x = i - kCenterX
            y = j - kCenterY
            val = GaussianValue(x, y, var)
            kernel[i, j] = val
            total += val
    kernel = kernel/total
    return kernel

def GaussianFilter(image, kSizeX, kSizeY, var):
    print("```Gaussian Filter```")
    kernel = MyGaussianKernel(kSizeX, kSizeY, var)
    Result = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    kCenterX = kSizeX//2
    kCenterY = kSizeY//2

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # for m in range(3):
            num = 0.0
            for k in range(kSizeX):
                for l in range(kSizeY):
                    row = i - kCenterX + k
                    col = j - kCenterY + l
                    # print(row, col)
                    if ( row >= 0 and row < image.shape[0] and col >= 0 and col < image.shape[1] ):
                        num += image[row, col]*kernel[k ,l]
            Result[i, j] = num
    return Result

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
