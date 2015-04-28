import numpy as np
import cv2
import argparse

# Prepare args
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to the input image")
ap.add_argument("-q", "--query", required=True, help="Path to the input image")
args = vars(ap.parse_args())

templatePath = args['template']
queryPath = args['query']

img1 = cv2.imread(queryPath,0)          # queryImage
img2 = cv2.imread(templatePath,0) # trainImage

# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,flags=2)

#plt.imshow(img3),plt.show()
cv.imshow('x', img3)
