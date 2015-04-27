import numpy as np
import cv2
import imutils
import argparse
import sys
#from matplotlib import pyplot as plt

print ''
print '-----------------------'
print 'Using OpenCV : ' + cv2.__version__
print '-----------------------'
print ''

# Prepare args
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to the input image")
ap.add_argument("-q", "--query", required=True, help="Path to the input image")
ap.add_argument("-he", "--hessian", help="Sorting method", default=400, type=float)
ap.add_argument("-o", "--output", help="Sorting method", default="data/out.jpg")
args = vars(ap.parse_args())

# Get args
templatePath = args['template']
queryPath = args['query']
hessian = float(args['hessian'])

# Load images
imgTemplate = cv2.imread(templatePath, 0)
imgQuery = cv2.imread(queryPath, 0)

# Create SURF
surf = cv2.SURF(hessian)
surf.upright = True
surf.hessianThreshold = hessian

# Detect
kp1, des1 = surf.detectAndCompute(imgTemplate,None)
kp2, des2 = surf.detectAndCompute(imgQuery,None)

# Matcher
bf = cv2.BFMatcher()

# Matches and sort them in the order of their distance.
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(imgTemplate,kp1,imgQuery,kp2,matches[:10], flags=2)
#plt.imshow(img3),plt.show()
cv2.imwrite(args['output'], img3)