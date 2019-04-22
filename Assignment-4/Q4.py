# Kaustav Vats (2016048)

import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import PanoramaStiching

PATH = "./Data/Q4/"
SAVE = "./Data/Q4-output/"

# Image Reading --------------------------

name = ["1a.jpg", "1b.jpg", "1c.jpg"]

Image1 = cv2.imread(PATH + name[0])
Image2 = cv2.imread(PATH + name[1])
Image3 = cv2.imread(PATH + name[2])

print("Image{}.shape: {}".format(1, Image1.shape))
print("Image{}.shape: {}".format(2, Image2.shape))
print("Image{}.shape: {}".format(3, Image3.shape))

# ----------------------------------------
new_img = np.zeros((Image2.shape[0], Image1.shape[1]+Image2.shape[1]+Image3.shape[1], 3), dtype=np.uint8)
new_img[0:Image2.shape[0], Image1.shape[1]:Image1.shape[1]+Image2.shape[1]] = Image2
NewImage2 = new_img
cv2.imwrite("./Data/Q4-output/NewImage2.png", NewImage2)

# Panarama Stiching ----------------------
FinalImageSize = NewImage2.shape
copySize = Image1.shape[1]

PS = PanoramaStiching()

Result1, Visualization1 = PS.StichImages(Image1, NewImage2, FinalImageSize)
Result1 = PS.Format(Result1, Image2, copySize)
cv2.imwrite(SAVE + "result1.png", Result1)
cv2.imwrite(SAVE + "Visualization1.png", Visualization1)

# Result2, Visualization2 = PS.StichImages(Image3, Image2, FinalImageSize)
# cv2.imwrite(SAVE + "result2.png", Result2)
# cv2.imwrite(SAVE + "Visualization2.png", Visualization2)

copySize = Image1.shape[1] + Image2.shape[1]
Result3, Visualization3 = PS.StichImages(Image3, Result1, FinalImageSize)
Result3 = PS.Format2(Result3, Result1, copySize)
cv2.imwrite(SAVE + "result3.png", Result3)
cv2.imwrite(SAVE + "Visualization3.png", Visualization3)

