#  Kaustav Vats
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle

# Ref: Bounding Box https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
# Ref: Feature matching etc. https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
# Ref Features Matching:- https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
# Ref: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html

LOAD_PATH = "./Data/Q3/"
SAVE_PATH = "./Data/Q3-output/"

def KeyPointsMatch(img1, img2):
    kp1, des1 = DetectKeyPoints(img1)
    kp2, des2 = DetectKeyPoints(img2)
    # sift = cv2.xfeatures2d.SIFT_create()

    # kp1, des1 = sift.detectAndCompute(img1, None)
    # kp2, des2 = sift.detectAndCompute(img2, None)

    Matches = MatchKepPoints(img1, img2, kp1, kp2, des1, des2)
    FinalMatches, Homography, Status = Matches


    result = DrawMatches(img1, img2, kp1, kp2, FinalMatches, Status)
    cv2.imwrite(SAVE_PATH + "Matched.png", result)

def DetectKeyPoints(img):
    sift = cv2.xfeatures2d.SIFT_create()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(img, None)

    FinalKps = []
    for keypoint in kp:
        FinalKps.append(keypoint.pt)
    FinalKps = np.float32(FinalKps)
    # print("FinalKps: ", FinalKps)
    return FinalKps, des

def DrawMatches(img1, img2, kp1, kp2, Matches, Status):
    (x1, y1) = img1.shape[:2]
    (x2, y2) = img2.shape[:2]
    result = np.zeros((max(x1, x2), y1 + y2, 3), dtype="uint8")
    result[0:x1, 0:y1] = img1
    result[0:x2, y1:] = img2

    for i in range(Status.shape[0]):
        if Status[i] != 1:
            Point1 = (int(kp1[Matches[i, 1]][0]), int(kp1[Matches[i, 1]][1]))
            Point2 = (int(kp2[Matches[i, 0]][0]) + y1, int(kp2[Matches[i, 0]][1]))        
            cv2.line(result, Point1, Point2, (0, 0, 255), 1)
        else:
            Point1 = (int(kp1[Matches[i, 1]][0]), int(kp1[Matches[i, 1]][1]))
            Point2 = (int(kp2[Matches[i, 0]][0]) + y1, int(kp2[Matches[i, 0]][1]))        
            cv2.line(result, Point1, Point2, (0, 255, ), 1)
    return result

def findHomography(P1, P2):

    n = P1.shape[0]
    A = np.zeros((2*n, 8))

    for i in range(n):
        r1 = [P1[i][0], P1[i][1], 1, 0, 0, 0, -P1[i][0]*P2[i][0], -P1[i][1]*P2[i][0]]
        r1 = [0, 0, 0, P1[i][0], P1[i][1], 1, -P1[i][0]*P2[i][1], -P1[i][1]*P2[i][1]]
    
    B = np.zeros((2*n, 1))
    for i in range(n):
        B[2*i] = P2[i][0]
        B[2*i + 1] = P2[i][1] 
    
    Inverse = np.linalg.inv(np.matmul(A.T, A))
    bz = np.matmul(A.T, B)
    homo = np.matmul(Inverse, bz)
    homo = np.append(homo, 1)
    homo = np.matmul(homo,(3,3))
    return homo

def MatchKepPoints(img1, img2, kp1, kp2, des1, des2, ratio=0.75, RansacThreshold=4.0):
    BF = cv2.BFMatcher()
    Matches = BF.knnMatch(des1, des2, k=2)

    # Ratio Test for point matching
    # good = []
    # for m,n in Matches:
    #     if m.distance < 0.7*n.distance:
    #         good.append(m)

    # if len(good)>10:
    #     Points1 = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #     Points2 = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    # FinalMatches = np.asarray()
    FinalMatches = []
    for i in range(len(Matches)):
        if Matches[i][0].distance < Matches[i][1].distance * ratio:
            FinalMatches.append((Matches[i][0].trainIdx, Matches[i][0].queryIdx))

    FinalMatches = np.asarray(FinalMatches)
    
    if FinalMatches.shape[0] > 4:
        Points1 = []
        Points2 = []
        for i in range(FinalMatches.shape[0]):
            Points1.append(kp1[FinalMatches[i, 1]])
            Points2.append(kp2[FinalMatches[i, 0]])
        
        Points1 = np.float32(Points1)
        Points2 = np.float32(Points2)

        Homography, status = cv2.findHomography(Points1, Points2, cv2.RANSAC, RansacThreshold)
        # Homography  = findHomography(Points1, Points2)
        print(Homography)

        return good, Homography, status
    return None

    # def DrawBox(self, )
    
if __name__ == "__main__":
    img1 = cv2.imread(LOAD_PATH + "collage.jpg")
    img2 = cv2.imread(LOAD_PATH + "test2.jpeg")

    # Mat = Matcher()
    KeyPointsMatch(img1, img2)

    # sift = cv2.xfeatures2d.SIFT_create()

    # kp1, des1 = sift.detectAndCompute(img1, None)
    # kp2, des2 = sift.detectAndCompute(img2, None)

    # Matches = MatchKepPoints(img1, img2, kp1, kp2, des1, des2)
    # FinalMatches, Homography, Status = Matches

    # result = DrawMatches(img1, img2, kp1, kp2, FinalMatches, Status)

    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks = 50)

    # flann = cv2.FlannBasedMatcher(index_params, search_params)

    # matches = flann.knnMatch(des1,des2,k=2)

    # good = []
    # for m,n in matches:
    #     if m.distance < 0.7*n.distance:
    #         good.append(m)

    # if len(good)>10:
    #     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    #     matchesMask = mask.ravel().tolist()

    #     h,w = img1.shape
    #     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #     dst = cv2.perspectiveTransform(pts,M)

    #     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    # else:
    #     matchesMask = None

    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                singlePointColor = None,
    #                matchesMask = matchesMask, # draw only inliers
    #                flags = 2)

    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    # cv2.imwrite(SAVE_PATH + "box2.png", img3)


