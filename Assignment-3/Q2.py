# Kaustav Vats (2016048)

import numpy as np
import cv2
from helper import *
from sklearn.cluster import MeanShift, estimate_bandwidth

# Image Reading ----------------------------------------------------------------------------

ImageList = ["iceCream1.jpg", "iceCream2.jpg", "iceCream3.jpg", "colors.jpg"]
No = 1
Image = cv2.imread("./Q2-images/"+ImageList[No])

# Reshape Image -------------------------------------------------------------
Image2 = np.reshape(Image, [-1, 3])
print("New Shape", Image2.shape)

# Find Clusters --------------------------------------------------------------
bandwidth = estimate_bandwidth(Image2, quantile=0.2, n_samples=500)
MS = MeanShift(bandwidth, bin_seeding=True)
MS.fit(Image2)

# Predict Labels -------------------------------------------------------------
labels = MS.labels_
print(labels.shape)
print("Clusters", MS.cluster_centers_)

Image2 = np.reshape(labels, [Image.shape[0], Image.shape[1]])
cv2.imwrite("./Q2-output/" + ImageList[No], Image2)

plt.figure()
plt.imshow(Image2)
plt.axis('off')
plt.show()


