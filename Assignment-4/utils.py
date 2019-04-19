# Kaustav Vats (2016048)
# Ref Panorama Stiching:- https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
# Ref Features Matching:- https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html

import cv2
import numpy as np
from matplotlib import pyplot as plt

class PanoramaStiching:

    def __init__(self):
        return 

    # def combine_images(self, img0, img1, h_matrix):
    #     points0 = np.array([[0, 0], [0, img0.shape[0]], [img0.shape[1], img0.shape[0]], [img0.shape[1], 0]], dtype=np.float32)
    #     points0 = points0.reshape((-1, 1, 2))
    #     points1 = np.array([[0, 0], [0, img1.shape[0]], [img1.shape[1], img0.shape[0]], [img1.shape[1], 0]], dtype=np.float32)
    #     points1 = points1.reshape((-1, 1, 2))
    #     points2 = cv2.perspectiveTransform(points1, h_matrix)
    #     points = np.concatenate((points0, points2), axis=0)
    #     [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
    #     [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)
    #     H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    #     output_img = cv2.warpPerspective(img1, H_translation.dot(h_matrix), (x_max - x_min, y_max - y_min))
    #     output_img[-y_min:img0.shape[0] - y_min, -x_min:img0.shape[1] - x_min] = img0
    #     return output_img

    def StichImages(self, img1, img2):

        kp1, des1 = self.DetectKeyPoints(img1)
        kp2, des2 = self.DetectKeyPoints(img2)

        Matches = self.MatchKepPoints(kp1, kp2, des1, des2)

        if Matches is None:
            return None, None

        FinalMatches, Homography, Status = Matches

        # Result = self.combine_images(img1, img2, Homography)

        # stiching the images based on the Homography
        Result = cv2.warpPerspective(img1, Homography, (img1.shape[1] + img2.shape[1], max(img1.shape[0], img2.shape[0])))
        # cv2.imshow("Result", Result)
        # cv2.waitKey(0)
        # for i in range(img2.shape[0]):
        #     for j in range(img2.shape[1]):
        #         if img2[i, j, 0] == 0 and img2[i, j, 1] == 0 and img2[i, j, 2] == 0:
        #             continue
        #         Result[i, j, 0] = img2[i, j, 0]
        #         Result[i, j, 1] = img2[i, j, 1]
        #         Result[i, j, 2] = img2[i, j, 2]

        # Result = cv2.resize(Result, None, fx=Result.shape[0], fy=Result.shape[1], interpolation = cv2.INTER_CUBIC)
        Result[0 : img2.shape[0], 0 : img2.shape[1]] = img2

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


