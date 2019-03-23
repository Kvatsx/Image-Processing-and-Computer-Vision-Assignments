# Kaustav Vats (2016048)

import numpy as np
import cv2
from helper import *

# ReadImage ------------------------------------------------------------------------
ImageNames = ["colors.jpg", "2apples.jpg", "2or4objects.jpg", "7apples.jpg", "variableObjects.jpeg"]
No = 1
Image = cv2.imread("./Q1-images/" + ImageNames[No])
print("Image Size:", Image.shape)
# Run K-Mean Cluster ----------------------------------------------------------------

KMC = KMeanCluster(Image, k=5, max_iterations=30, filename="./Q1-output/" + ImageNames[No])
KMC.run()

KMC5 = KMeanCluster5D(Image, k=5, max_iterations=30, filename="./Q1-output/5D_" + ImageNames[No])
KMC5.run()


# Q3 https://arxiv.org/ftp/arxiv/papers/1708/1708.02694.pdf