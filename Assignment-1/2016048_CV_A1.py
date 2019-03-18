# @Author: Kaustav Vats 
# @Roll-Number: 2016048 

# In[]
from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
from random import randint as randi
import math
import pywt

# Util Functions

def SaltAndPepperNoise(image, Percentage):
    Pixels = (image.shape[0]*image.shape[1] * Percentage)//100
    # print(Pixels, image.shape[0]*image.shape[1])
    pairs = []
    i=0
    Result = np.copy(image)

    while i < Pixels:
        x = randi(0, image.shape[0]-1)
        y = randi(0, image.shape[1]-1)
        z = randi(0, 255)
        if (x, y) not in pairs:
            Result[x, y] = z
            pairs.append((x, y))
        else:
            continue
        i += 1
    return Result

def ShowResults(original, result, title):
    plt.subplot(121),plt.imshow(original),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(result),plt.title(title)
    plt.xticks([]), plt.yticks([])


def AvgFilter(Image, kSizeX, kSizeY):
    Result = np.zeros((Image.shape[0], Image.shape[1]), dtype=np.uint8)
    kCenterX = kSizeX//2
    kCenterY = kSizeY//2

    for i in range(Image.shape[0]):
        for j in range(Image.shape[1]):
            num = 0
            for k in range(kSizeX):
                for l in range(kSizeY):
                    row = i - kCenterX + k
                    col = j - kCenterY + l
                    # print(row, col)
                    if ( row >= 0 and row < Image.shape[0] and col >= 0 and col < Image.shape[1] ):
                        num += Image[row, col]
            Result[i, j] = num/(kSizeX * kSizeY)
    return Result

def MedianFilter(image, kSizeX, kSizeY):
    Result = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    kCenterX = kSizeX//2
    kCenterY = kSizeY//2

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            num = []
            for k in range(kSizeX):
                for l in range(kSizeY):
                    row = i - kCenterX + k
                    col = j - kCenterY + l
                    # print(row, col)
                    if ( row >= 0 and row < image.shape[0] and col >= 0 and col < image.shape[1] ):
                        num.append(image[row, col])
            num.sort()
            Result[i, j] = num[len(num)//2]
    return Result


def GaussianValue(x, y, var=1.0):
    s = 2.0*var*var
    val = math.exp(-((x*x+y*y) /s ))
    den = math.pi * s
    return val/den

def MyGaussianKernel(kSizeX, kSizeY, var):
    kernel = np.zeros((kSizeX, kSizeY))
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

def GaussianFilter(image, kernel, kSizeX, kSizeY):
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

def SuppressHighFrequency(arr):
    
    for k in range(len(arr[1])):
        for i in range(len(arr[1][k])):
            for j in range(len(arr[1][k][i])):
                arr[1][k][i][j] = 0
        
        for i in range(len(arr[2][2])):
            for j in range(len(arr[2][k][i])):
                arr[2][k][i][j] = 0

    return arr

def ApplyWatermark(arr, brr, a=0.5):
    print(len(arr[0]), len(brr[0]))
    print(len(arr[0][1]), len(brr[0][1]))    
    for i in range(len(arr[0])):
        for j in range(len(arr[0][i])):
            arr[0][i][j] = arr[0][i][j] + a*brr[0][i][j]
    return arr

def DownsampleMyImage(image):
    result = np.zeros((image.shape[0]//2, image.shape[1]//2), dtype=np.uint8)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            x = i*2
            y = j*2
            num = image[x, y]
            # num += image[x, y+1]
            # num += image[x+1, y]
            # num += image[x+1, y+1]
            # result[i, j] = num/4
            result[i, j] = num
    return result

def Addition(img1, img2):
    result = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if int(img1[i, j]) + int(img2[i, j]) > 255:
                result[i, j] = 255
            else:
                result[i, j] = img1[i, j] + img2[i, j]
    return result


def CalculateDiff(img1, img2):
    result = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)
    count = 0
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if ( int(img1[i, j]) - int(img2[i, j]) < 0 ):
                # result[i, j] = abs(img1[i, j] - img2[i, j])
                # print("print krle", type(img1[i-j]-img2[i, j]))
                result[i, j] = img2[i, j] - img1[i, j]
            else:
                result[i, j] = img1[i, j] - img2[i, j]
            if result[i, j] > 0:
                count += 1
    print(count, img1.shape[0], img1.shape[1])
    # print(count*100)/float(int(img1.shape[0])*int(img1.shape[1]))
    return result

def UpscaleMyImage(img):
    result = np.zeros((img.shape[0]*2, img.shape[1]*2), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i*2 ,j*2] = img[i, j]
    return result

def Interpolation(img):
    kernel = np.asarray([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])
    print(kernel)

    Result = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            num = 0.0
            for k in range(3):
                for l in range(3):
                    row = i - 1 + k
                    col = j - 1 + l
                    # print(row, col)
                    if ( row >= 0 and row < img.shape[0] and col >= 0 and col < img.shape[1] ):
                        num += kernel[k, l] * img[row, col]
            Result[i, j] = num
    return Result

def LaplacianFilter(image):
    kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    kernel = np.asarray(kernel)

    Result = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    kCenterX = 1
    kCenterY = 1

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
    return Result

def AbsDiff(img1, img2):
    result = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if ( int(img1[i, j]) - int(img2[i, j]) < 0 ):
                # result[i, j] = abs(img1[i, j] - img2[i, j])
                # print("print krle", type(img1[i-j]-img2[i, j]))
                result[i, j] = img2[i, j] - img1[i, j]
            else:
                result[i, j] = img1[i, j] - img2[i, j]
    return result

# Q1 ------------------------
# In[]
Image = cv2.imread('image_1.jpg', 0) # 1, 0, -1
print("Image.shape", Image.shape)

resultImage = AvgFilter(Image, 15, 15)
print(resultImage)
# cv2.imshow('myimage', resultImage)
ShowResults(Image, resultImage, "Blurred")

cv2.imwrite("original.png", Image)
cv2.imwrite("blur.png", resultImage)
# cv2.imshow("result", resultImage)
# cv2.waitKey(0)


# Q2 ------------------------
# In[]

Image2 = cv2.imread('image_2.png', 0)
print("Image.shape", Image2.shape)

result = SaltAndPepperNoise(Image2, 10)

# In[]
cv2.imwrite("original.png", Image2)
cv2.imwrite("SaltAndPepper.png", result)

result2 = MedianFilter(result, 11, 11)
cv2.imwrite("MedianFilter.png", result2)

# Q3 ------------------------

# In[]

kSize = 11
Image3 = cv2.imread('image_3.png', 0)
print(Image3.shape)
kernel = MyGaussianKernel(kSize, kSize, 3)
print(kernel)
result = GaussianFilter(Image3, kernel, kSize, kSize)

cv2.imwrite("original.png", Image3)
cv2.imwrite("Gaussian.png", result)



# Q4 ------------------------
# In[]
# 3 Level Gaussian

Image3 = cv2.imread("image_3.png", 0)
kSize = 5
kernel = MyGaussianKernel(kSize, kSize, 5)
result = Image3
for i in range(3):
    result = GaussianFilter(result, kernel, kSize, kSize)
    cv2.imwrite("result.png", result)
    # result = cv2.resize(result, (result.shape[1]//2, result.shape[0]//2), interpolation = cv2.INTER_AREA)
    result = DownsampleMyImage(result)
    ShowResults(Image3, result, "Downsample " + str(i))
    cv2.imwrite("Downscale"+str(i)+".png", result)

for i in range(2, -1, -1):
    lap_img = cv2.imread("Downscale"+str(i)+".png", 0)
    lap_up = UpscaleMyImage(lap_img)
    cv2.imwrite("lap_up.png", lap_up)
    lap_up = Interpolation(lap_up)
    cv2.imwrite("lap_up2.png", lap_up)
    if i == 0:
        gauss = Image3
    else:
        gauss = cv2.imread("Downscale"+str(i-1)+".png", 0)
    sub_res = CalculateDiff(gauss, lap_up)
    cv2.imwrite("lap"+str(i)+".png", sub_res)


# Q5 ------------------------
# In[]
# Part1
Image = cv2.imread('image_1.jpg', 0) 

resultImage = AvgFilter(Image, 15, 15)
cv2.imwrite("original.png", Image)
cv2.imwrite("blur.png", resultImage)

avg = cv2.blur(Image, (15, 15))
cv2.imwrite("avg.png", avg)

diff = CalculateDiff(resultImage, avg)
cv2.imwrite("diff1.png", diff)
diff2 = cv2.subtract(resultImage, avg)
cv2.imwrite("diff2.png", diff2)

# In[]
# Part2
Image2 = cv2.imread('image_2.png', 0)
result = SaltAndPepperNoise(Image2, 10)
# cv2.imwrite("original.png", Image2)
cv2.imwrite("SaltAndPepper.png", result)
result2 = MedianFilter(result, 11, 11)
cv2.imwrite("MedianFilter.png", result2)

med = cv2.medianBlur(result, 11)
cv2.imwrite("Med.png", med)
diff = CalculateDiff(result2, med)
diff2 = cv2.subtract(result2, med)
cv2.imwrite("diff2.png", diff2)
cv2.imwrite("diff.png", diff)


# In[]
# Part3
kSize = 15
Image3 = cv2.imread('image_3.png', 0)
kernel = MyGaussianKernel(kSize, kSize, 5)
result = GaussianFilter(Image3, kernel, kSize, kSize)
cv2.imwrite("original.png", Image3)
cv2.imwrite("Gaussian.png", result)

gb = cv2.GaussianBlur(Image3, (15, 15), sigmaX=5, sigmaY=5)
cv2.imwrite("gb.png", gb)

diff2 = cv2.subtract(result, gb)
cv2.imwrite("diff2.png", diff2)

diff = CalculateDiff(result, gb)
cv2.imwrite("gb2.png", diff)


# Q6 ------------------------
# In[]

Image3 = cv2.imread('image_3.png', 0)
cv2.imwrite("original.png", Image3)
noise = SaltAndPepperNoise(Image3, 10)
cv2.imwrite("Noise.png", noise)
# ShowResults(Image3, noise, "Noise")
Laplacian_img = LaplacianFilter(Image3)
# Laplacian_img = cv2.subtract(Image3, Laplacian_img)
# Laplacian_image = cv2.Laplacian(Image3, cv2.CV_8U)
cv2.imwrite("LapNew.png", Laplacian_img)
# cv2.imwrite("Lap.png", Laplacian_image)
# ShowResults(Image3, Laplacian_image, "LaplacianImage")

# result = cv2.add(Image3, noise)
# # print(result.shape)
# # print(Laplacian_image.shape)
# result = cv2.add(result, Laplacian_img)
# # ShowResults(Image3, result, "Result")
# cv2.imwrite("result.png", result)

r2 = Addition(noise, Laplacian_img)
cv2.imwrite("r2.png", r2)


# In[]
# haar
dec2 = pywt.wavedec2(r2, 'haar', level=2)
print(len(dec2))

arr = SuppressHighFrequency(dec2)
# print(arr, len(arr), len(arr[1]), len(arr[2]))
result = pywt.waverec2(arr, 'haar')
cv2.imwrite("result2.png", result)



# Q7 ------------------------
# In[]
Image3 = cv2.imread('image_3.png', 0)
Watermark = cv2.imread('watermark.png', 0)
print(Image3.shape)
print(Watermark.shape)
Watermark = cv2.resize(Watermark, (Image3.shape[1], Image3.shape[0]), interpolation = cv2.INTER_AREA)
print(Watermark.shape)

cv2.imwrite("original.png", Image3)
cv2.imwrite("New_watermark.png", Watermark)

dec3 = pywt.wavedec2(Image3, 'db1', level=3)
wdec3 = pywt.wavedec2(Watermark, 'db1', level=3) 
print(len(dec3))

dec3 = ApplyWatermark(dec3, wdec3, 0.08)
result= pywt.waverec2(dec3, 'db1')
cv2.imwrite("WaterMarkResult.png", result)




# In[]