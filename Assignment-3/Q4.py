# Kaustav Vats (2016048)

import numpy as np
import cv2
from helper import *
from skimage.feature import hog, local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import pickle

# Extract Features -----------------------------------------------------------
class BagOfVisualWords:

    def HOG_features(self, x_train):
        fds = []
        for i in range(x_train.shape[0]):
            Image = np.reshape(x_train[i], (32, 32, 3))
            # cv2.imwrite("./Q4-output/original.png", Image)
            fd, _ = hog(Image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True, multichannel=True)
            # cv2.imwrite("./Q4-output/hog1.png", hog_image)
            fds.append(np.reshape(fd, (1, fd.shape[0])))
            fds.append(fd)
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            # ax1.axis('off')
            # ax1.imshow(Image, cmap=plt.cm.gray)
            # ax1.set_title('Input image')
        for i in range(1, len(fds)):
            fds[0] = np.concatenate((fds[0], fds[i]))
        fds[0] = np.reshape(fds[0], (fds[0].shape[0], 1))
        print("Fds.shape", fds[0].shape)
        return fds[0]

    def LBP_features(self, x_train, num_points = 24, radius = 8):
        fds = []
        for i in range(x_train.shape[0]):
            Image = np.reshape(x_train[i], (32, 32, 3))
            Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
            lbp = local_binary_pattern(Image, num_points, radius, method="uniform")
            # cv2.imwrite("./Q4-output/lbp1.png", lbp)
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
            hist = hist.astype(dtype=np.float)
            hist /= np.sum(hist)
            # fds.append(np.reshape(hist, (1, hist.shape[0])))
            fds.append(hist)
        for i in range(1, len(fds)):
            fds[0] = np.concatenate((fds[0], fds[i]))
        fds[0] = np.reshape(fds[0], (fds[0].shape[0], 1))
        print("Fds.shape", fds[0].shape)
        return fds[0]

    def getHistorgram(self, kmeans, x_data, num_points = 24, radius = 8, hog_flag=True):
        features = []
        hist_size = len(kmeans.cluster_centers_)
        print("Clusters:", hist_size)
        for i in range(x_data.shape[0]):
            Image = np.reshape(x_data[i], (32, 32, 3))
            histogram = np.zeros((hist_size, ), dtype=np.int)
            if hog_flag:
                fd, _ = hog(Image, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True, multichannel=True)
                print(fd.shape)
                fd = np.reshape(fd, (fd.shape[0], 1))
                print(kmeans.predict(fd))
                histogram[kmeans.predict(fd)] += 1
            else:
                Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
                lbp = local_binary_pattern(Image, num_points, radius, method="uniform")  # cv2.imwrite("./Q4-output/lbp1.png", lbp)
                hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
                hist = hist.astype(dtype=np.float)
                hist /= np.sum(hist)
                hist = np.reshape(fd, (hist.shape[0], 1))
                histogram[kmeans.predict(hist)] += 1
            features.append(histogram)
        features = np.asarray(features)
        return features


    def run(self, x_train, y_train, x_test, y_test, hog_flag=True):
        fd_list = []
        if hog_flag:
            fd_list = self.HOG_features(x_train)
        else:
            fd_list = self.LBP_features(x_train)
        print("------------------Features Collected-----------------")

        kmeans = KMeans(n_clusters = 800)
        kmeans.fit(fd_list)
        pickle.dump(kmeans, open("./Q4-output/kmeans.sav", 'wb'))
        print("------------------K-Mean Clutering Done--------------")

        # kmeans = pickle.load(open("./Q4-output/kmeans.sav", 'rb'))
        nx_train = self.getHistorgram(kmeans, x_train, hog_flag=hog_flag)
        print("nx_train.shape", nx_train.shape)
        nx_test = self.getHistorgram(kmeans, x_test, hog_flag=hog_flag)
        print("nx_test.shape", nx_test.shape)
        print("------------------Histogram Features-----------------")

        print("------------------Training SVM-----------------------")
        clf = SVC(gamma='scale', decision_function_shape='ovo')
        clf.fit(nx_train, y_train)

        print("------------------Predicting Score-------------------")
        Accuracy = clf.score(nx_test, y_test)

        print("Acc", Accuracy*100)



# Read Data -------------------------------------------------------------------

X_Train, Y_Train, X_Test, Y_Test = ReadData()

Bovw = BagOfVisualWords()
Bovw.run(X_Train, Y_Train, X_Test, Y_Test, hog_flag=False)
