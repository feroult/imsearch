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
ap.add_argument("-if", "--image-first", required=True, help="Path to the input image")
ap.add_argument("-is", "--image-second", required=True, help="Path to the input image")
ap.add_argument("-he", "--hessian", required=False, help="Sorting method")
args = vars(ap.parse_args())

# Get args
firstImage = args['image-first']
secondImage = args['image-second']
hessian = float(args['hessian'])

# Create SURF
surf = cv2.SURF(hessian)
surf.upright = True
surf.hessianThreshold = hessian

# Detect
kp, des = surf.detectAndCompute(gray,None)