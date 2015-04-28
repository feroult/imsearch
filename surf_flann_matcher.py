import numpy as np
import cv2
import imutils
import argparse
import sys

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



def drawMatches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imwrite(args['output'], out)



def flaanMatcher():
	# FLANN parameters
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)

	# Matcher
	flann = cv2.FlannBasedMatcher(index_params,search_params)

	# Matches and sort them in the order of their distance.
	matches = flann.knnMatch(des1,des2,k=2)

	# Need to draw only good matches, so create a mask
	matchesMask = [[0,0] for i in xrange(len(matches))]

	# ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
	    if m.distance < 0.7*n.distance:
	        matchesMask[i]=[1,0]

	draw_params = dict(matchColor = (0,255,0),
	                   singlePointColor = (255,0,0),
	                   matchesMask = matchesMask,
	                   flags = 0)

	img3 = drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params)





def simpleMatcher(img1, img2, kp1, kp2, des1, des2):
	# create BFMatcher object
	bf = cv2.BFMatcher()

	# Matches and sort them in the order of their distance.
	matches = bf.match(des1,des2)
	matches = sorted(matches, key = lambda x:x.distance)

	max_dist = 0
	min_dist = 100

	for x in xrange(0, des1.itemsize):
		dist = matches[x].distance

		if dist < min_dist:
			min_dist = dist

    	if dist > max_dist:
    		max_dist = dist

	drawMatches(img1, kp1, img2, kp2, matches)




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

# Using simple matcher
simpleMatcher(imgTemplate, imgQuery, kp1, kp2, des1, des2)