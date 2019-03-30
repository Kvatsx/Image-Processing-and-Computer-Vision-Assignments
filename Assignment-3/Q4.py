# Kaustav Vats (2016048)

import numpy as np
import cv2
from helper import *
from skimage.feature import hog, local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pickle
from random import randint as randi

# Extract Features -----------------------------------------------------------
def HOG_features(x_train):
    fds = []
    for i in range(x_train.shape[0]):
        Image = np.reshape(x_train[i], (32, 32, 3))
        # cv2.imwrite("./Q4-output/original.png", Image)
        # fd, _ = hog(Image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True, multichannel=True, block_norm="L2-Hys")
        fd, _ = hog(Image, visualize=True, multichannel=True, block_norm="L2-Hys")

        fds.extend(fd)

    fds = np.asarray(fds)
    fds = np.reshape(fds, (fds.shape[0], 1))
    print("Fds.shape", fds.shape)
    return fds

def LBP_features(x_train, num_points = 24, radius = 8):
    fds = []
    for i in range(x_train.shape[0]):
        Image = np.reshape(x_train[i], (32, 32, 3))
        Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(Image, num_points, radius, method="uniform")
        # cv2.imwrite("./Q4-output/lbp1.png", lbp)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
        hist = hist.astype(dtype=np.float)
        hist /= np.sum(hist)
        fds.extend(hist)
    
    fds = np.asarray(fds)
    fds = np.reshape(fds, (fds.shape[0], 1))
    print("Fds.shape", fds.shape)
    return fds

def getHistorgram(kmeans, x_data, num_points = 24, radius = 8, hog_flag=True):
    features = []
    hist_size = len(kmeans.cluster_centers_)
    # print("Clusters:", hist_size)
    for i in range(x_data.shape[0]):
        Image = np.reshape(x_data[i], (32, 32, 3))
        histogram = np.zeros((hist_size, ))
        if hog_flag:
            # fd, _ = hog(Image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True, multichannel=True, block_norm="L2-Hys")
            fd, _ = hog(Image, visualize=True, multichannel=True, block_norm="L2-Hys")
            norm = np.size(fd)
            fd = np.reshape(fd, (fd.shape[0], 1))
            histogram[kmeans.predict(fd)] += 1/norm
        else:
            Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
            lbp = local_binary_pattern(Image, num_points, radius, method="uniform")  # cv2.imwrite("./Q4-output/lbp1.png", lbp)
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
            hist = hist.astype(dtype=np.float)
            hist /= np.sum(hist)
            hist = np.reshape(hist, (hist.shape[0], 1))
            histogram[kmeans.predict(hist)] += 1
        features.append(histogram)
    features = np.asarray(features)
    return features

def Visualize(x_train):
    for i in range(5):
        obj = x_train[randi(0, x_train.shape[0]-1)]
        Image = np.reshape(obj, (32, 32, 3))
        cv2.imwrite("./Q4-output/orig_"+str(i)+".png", Image)
        # cv2.imwrite("./Q4-output/original.png", Image)
        # fd, _ = hog(Image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True, multichannel=True, block_norm="L2-Hys")
        fd, hogimg = hog(Image, visualize=True, multichannel=True, block_norm="L2-Hys")
        plt.figure()
        n, bins, patches = plt.hist(x=fd, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('File Descriptor count')
        plt.ylabel('Frequency')
        plt.title('Histogram of Gradients')
        plt.savefig("./Q4-output/hoghist_"+str(i)+".png")
        cv2.imwrite("./Q4-output/hog_"+str(i)+".png", hogimg)
        Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(Image, 12, 3)
        # cv2.imwrite("./Q4-output/lbp1.png", lbp)
        histo, lbpigm = np.histogram(lbp.ravel(), bins=np.arange(0, 12 + 3), range=(0, 3 + 2))
        cv2.imwrite("./Q4-output/lbp_"+str(i)+".png", lbp)

        plt.figure()
        n, bins, patches = plt.hist(x=histo, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Texture Range')
        plt.ylabel('Frequency')
        plt.title('Histogram of Texture')
        plt.savefig("./Q4-output/lbphist_"+str(i)+".png")

# Read Data -------------------------------------------------------------------

X_Train, Y_Train, X_Test, Y_Test = ReadData()
X_Train = X_Train[:10000]
Y_Train = Y_Train[:10000]
X_Test = X_Test[:1000]
Y_Test = Y_Test[:1000]

# fd_list = HOG_features(X_Train)
# fd_list = LBP_features(X_Train)
# hog_flag = False
# # print("------------------Features Collected-----------------")

# # kmeans = KMeans(n_clusters = 600)
# # kmeans.fit(fd_list)
# # pickle.dump(kmeans, open("./Q4-output/kmeans.sav", 'wb'))
# # print("------------------K-Mean Clutering Done--------------")

# kmeans = pickle.load(open("./Q4-output/kmeans.sav", 'rb'))
# nx_train = getHistorgram(kmeans, X_Train, hog_flag=hog_flag)
# print("nx_train.shape", nx_train.shape)
# nx_test = getHistorgram(kmeans, X_Test, hog_flag=hog_flag)
# print("nx_test.shape", nx_test.shape)
# print("------------------Histogram Features-----------------")

# print("------------------Training SVM-----------------------")
# clf = SVC(gamma='scale', decision_function_shape='ovo')
# clf.fit(nx_train, Y_Train)

# print("------------------Predicting Score-------------------")
# Accuracy = clf.score(nx_test, Y_Test)

# print("Acc", Accuracy*100)

Visualize(X_Train[:5])
