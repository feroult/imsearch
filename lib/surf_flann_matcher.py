import numpy as np
import cv2
import imutils
import argparse

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

print len(kp1)
print len(kp2)