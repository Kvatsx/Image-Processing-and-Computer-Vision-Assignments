# Kaustav Vats (2016048)
# Ref Panorama Stiching:- https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
# Ref Features Matching:- https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
import json
import torch

class PanoramaStiching:

    def __init__(self):
        return 

    # def RemoveBorder(self, image):
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    #     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     cnt = contours[0]
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     crop = image[y : y+h, x : x+w]
    #     return crop
    
    def Format(self, res, img, width):
        print(res.shape)
        print(width, img.shape[1])
        res[0:img.shape[0], width : width+img.shape[1]] = img
        return res

    def Format2(self, res, img, width):
        print(res.shape)
        print(width, img.shape[1])
        res[0:img.shape[0], 0 : width] = img[0:img.shape[0], 0 : width]
        return res

    def StichImages(self, img1, img2, resultSize):

        kp1, des1 = self.DetectKeyPoints(img1)
        kp2, des2 = self.DetectKeyPoints(img2)

        Matches = self.MatchKepPoints(kp1, kp2, des1, des2)

        if Matches is None:
            return None, None

        FinalMatches, Homography, Status = Matches

        # stiching the images based on the Homography
        Result = cv2.warpPerspective(img1, Homography, (resultSize[1], resultSize[0]))

        # Result = cv2.warpPerspective(img1, Homography, (img1.shape[1] + img2.shape[1], max(img1.shape[0], img2.shape[0])))
        cv2.imwrite("./Data/Q4-output/temp.png", Result)
        # cv2.imshow("Result", Result)
        # cv2.waitKey(0)
        # Result[0 : img2.shape[0], 0 : img2.shape[1]] = img2
        # Result = self.RemoveBorder(Result)
        
        StichedImage = self.DrawMatches(img1, img2, kp1, kp2, FinalMatches, Status)
        return Result, StichedImage

    def DetectKeyPoints(self, img):
        sift = cv2.xfeatures2d.SIFT_create()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(img, None)

        FinalKps = []
        for keypoint in kp:
            FinalKps.append(keypoint.pt)
        FinalKps = np.float32(FinalKps)
        # print("FinalKps: ", FinalKps)
        return FinalKps, des

    def MatchKepPoints(self, kp1, kp2, des1, des2, ratio=0.75, RansacThreshold=4.0):
        BF = cv2.BFMatcher()
        Matches = BF.knnMatch(des1, des2, k=2)

        # Ratio Test for point matching
        FinalMatches = []
        for i in range(len(Matches)):
            if Matches[i][0].distance < Matches[i][1].distance * ratio:
                FinalMatches.append((Matches[i][0].trainIdx, Matches[i][0].queryIdx))

        FinalMatches = np.asarray(FinalMatches)
        # print("FinalMatches.shape: {}".format(FinalMatches.shape))

        if FinalMatches.shape[0] > 4:
            Points1 = []
            Points2 = []
            for i in range(FinalMatches.shape[0]):
                Points1.append(kp1[FinalMatches[i, 1]])
                Points2.append(kp2[FinalMatches[i, 0]])
            
            Points1 = np.float32(Points1)
            Points2 = np.float32(Points2)
            # print("Point1.shape: {}".format(Points1.shape))
            # print("Point2.shape: {}".format(Points2.shape))

            Homography, status = cv2.findHomography(Points1, Points2, cv2.RANSAC, RansacThreshold)

            return FinalMatches, Homography, status
        return None
    
    def DrawMatches(self, img1, img2, kp1, kp2, Matches, Status):
        (x1, y1) = img1.shape[:2]
        (x2, y2) = img2.shape[:2]
        result = np.zeros((max(x1, x2), y1 + y2, 3), dtype="uint8")
        result[0:x1, 0:y1] = img1
        result[0:x2, y1:] = img2

        for i in range(Status.shape[0]):
            if Status[i] != 1:
                continue
            Point1 = (int(kp1[Matches[i, 1]][0]), int(kp1[Matches[i, 1]][1]))
            Point2 = (int(kp2[Matches[i, 0]][0]) + y1, int(kp2[Matches[i, 0]][1]))        
            cv2.line(result, Point1, Point2, (0, 255, 0), 1)
        
        return result


def ReadData():
    PATH = "./Data/"
    data = []
    label = []
    for i in range(1, 6):
        with open(PATH + "Q2/data_batch_" + str(i), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            data.append(dict[b'data'])
            label.append(dict[b'labels'])

    for i in range(1, 5):
        data[0] = np.concatenate((data[0], data[i]), axis = 0)
        label[0] = np.concatenate((label[0], label[i]), axis = 0)

    test = []
    test_label = []
    with open(PATH + "Q2/test_batch", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        test.append(dict[b'data'])
        test_label.append(dict[b'labels'])

    data = np.asarray(data[0])
    label = np.asarray(label[0])
    test = np.asarray(test[0])
    test_label = np.asarray(test_label[0])
    print("Train Data shape: {}".format(data.shape))
    print("Test Data shape: {}".format(test.shape))
    return data, label, test, test_label


def ReadLabels(className):
    PATH = "./Data/"
    LabelsNum = []
    with open(PATH + "Q2/batches.meta", 'rb') as data_file:
        data = pickle.loads(data_file.read())
        data = data['label_names']
        for i in range(len(data)):
            if data[i] in className:
                LabelsNum.append(i)
    LabelsNum = np.asarray(LabelsNum)
    return LabelsNum

def FilterData(data, labels, classNum):
    n_data = []
    n_label = []
    for i in range(labels.shape[0]):
        if labels[i] in classNum:
            n_data.append(data[i])
            n_label.append(labels[i])
    n_data = np.asarray(n_data)
    n_label = np.asarray(n_label)
    return n_data, n_label

def Turn2DataLoader(data, label):
    dataloader = []
    for i in range(data.shape[0]):
        img = np.reshape(data[i], (3, 32, 32))
        img = torch.from_numpy(img.astype(np.float64))
        dataloader.append((img, label[i]))
    return dataloader

