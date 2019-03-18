# Kaustav Vats (2016048)

import numpy as np
import cv2
import sys
from random import randint as randi
from tqdm import tqdm
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class KMeanCluster:
    Clusters = None
    Image = None
    Centroids = None
    MaxIterations = None

    def __init__(self, image, k=2, max_iterations=100):
        self.Clusters = k
        self.Image = image
        self.Centroids = np.zeros((k, 3), dtype=np.float64)
        self.MaxIterations = max_iterations

    def GetRandomColor(self):
        x = randi(0, self.Image.shape[0]-1)
        y = randi(0, self.Image.shape[1]-1)
        b = float(self.Image[x, y, 0])
        g = float(self.Image[x, y, 1])
        r = float(self.Image[x, y, 2])
        return b, g, r

    def RandomClusters(self):
        for i in range(self.Clusters):
            b, g, r = self.GetRandomColor()
            self.Centroids[i, 0] = b
            self.Centroids[i, 1] = g
            self.Centroids[i, 2] = r
    
    def GetEuclideanDistance(self, Cbgr, Ibgr):
        b = float(Cbgr[0]) - float(Ibgr[0])
        g = float(Cbgr[1]) - float(Ibgr[1])
        r = float(Cbgr[1]) - float(Ibgr[1])
        return sqrt(b*b + g*g + r*r)

    def bgr2hex(self, bgr):
        return "#%02x%02x%02x" % (int(bgr[2]), int(bgr[1]), int(bgr[0]))

    def ScatterPlot(self, clutserLabels):
        fig = plt.figure()
        ax = Axes3D(fig)
        for x in range(self.Image.shape[0]):
            for y in range(self. Image.shape[1]):
                ax.scatter(self.Image[x, y, 2], self.Image[x, y, 1], self.Image[x, y, 0], color=self.bgr2hex(self.Centroids[clutserLabels[x, y]]))
        plt.show()
        # plt.savefig("./Q1-output/ScatterPlot.png")

    def VisualizeCluster(self, clusterLabels):
        result = np.zeros((self.Image.shape), dtype=np.uint8)
        for i in range(self.Image.shape[0]):
            for j in range(self.Image.shape[1]):
                bgr = self.Centroids[clusterLabels[i, j]]
                result[i, j, 0] = np.uint8(bgr[0])
                result[i, j, 1] = np.uint8(bgr[1])
                result[i, j, 2] = np.uint8(bgr[2])
        cv2.imwrite("./Q1-output/kmeanResult.png", result)
        self.ScatterPlot(clusterLabels)
        # cv2.imshow("K-Mean Cluster", result)
        # cv2.waitKey(0)

    def run(self):
        self.RandomClusters()
        print("Centroids:\n", self.Centroids)
        ClusterLabels = np.zeros((self.Image.shape[0], self.Image.shape[1]), dtype=np.uint8)
        for i in tqdm(range(self.MaxIterations)):
            for x in range(self.Image.shape[0]):
                for y in range(self.Image.shape[1]):
                    MinDist = sys.float_info.max
                    for c in range(self.Clusters):
                        dist = self.GetEuclideanDistance(self.Centroids[c], self.Image[x, y])
                        if dist <= MinDist:
                            MinDist = dist
                            ClusterLabels[x, y] = c

            # update Mean of the clusters
            MeanCluster = np.zeros((self.Clusters, 4), dtype=np.float64)
            for x in range(self.Image.shape[0]):
                for y in range(self.Image.shape[1]):
                    clusterNumber = ClusterLabels[x, y]
                    MeanCluster[clusterNumber, 0] += 1
                    MeanCluster[clusterNumber, 1] += float(self.Image[x, y, 0])
                    MeanCluster[clusterNumber, 2] += float(self.Image[x, y, 1])
                    MeanCluster[clusterNumber, 3] += float(self.Image[x, y, 2])
            
            for c in range(self.Clusters):
                # print("MeanCluster["+ str(c) +", 0]:", MeanCluster[c, 0])
                self.Centroids[c, 0] = MeanCluster[c, 1] / MeanCluster[c, 0]
                self.Centroids[c, 1] = MeanCluster[c, 2] / MeanCluster[c, 0]
                self.Centroids[c, 2] = MeanCluster[c, 3] / MeanCluster[c, 0]
        
        self.VisualizeCluster(ClusterLabels)
    