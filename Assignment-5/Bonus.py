'''
 * @Author: Kaustav Vats 
 * @Roll-Number: 2016048 
 * @Date: 2019-04-26 23:34:33 
 * Ref:- [1] https://rdmilligan.wordpress.com/2016/05/23/disparity-of-stereo-images-with-python-and-opencv/
 * Ref:- [2] https://github.com/jagracar/OpenCV-python-tests/blob/master/OpenCV-tutorials/cameraCalibration/depthMap.py
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2

PATH = "./Data/"

Images = ["left.jpg", "right.jpg"]

Left = cv2.imread(PATH + Images[0], 0)
Right = cv2.imread(PATH + Images[1], 0)

# 32, 17
# StereoDepthMap =  cv2.StereoBM_create(numDisparities=32, blockSize=17)
window_size = 5
min_disp = 32
num_disp = 112-min_disp
StereoDepthMap = cv2.StereoSGBM(
    minDisparity = min_disp,
    numDisparities = num_disp,
    SADWindowSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 1,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    fullDP = False
)

DisparityMatrix = StereoDepthMap.compute(Left, Right)

min = DisparityMatrix.min()
max = DisparityMatrix.max()
DisparityMatrix = np.uint8(255 * (DisparityMatrix - min) / (max - min))

cv2.imwrite(PATH + "depth_map.png", DisparityMatrix)
