# Kaustav Vats (2016048)

import numpy as np
import cv2
from helper import *

# ReadImage ------------------------------------------------------------------------
ImageNames = ["colors.jpg", "2apples.jpg", "2or4objects.jpg", "7apples.jpg", "variableObjects.jpeg"]

Image = cv2.imread("./Q1-images/" + ImageNames[0])
print("Image Size:", Image.shape)
# Run K-Mean Cluster ----------------------------------------------------------------

KMC = KMeanCluster(Image, k=5, max_iterations=30)
KMC.run()
