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
ap.add_argument("-o", "--output", help="Sorting method", default="work/out.jpg")
ap.add_argument("-m", "--matcher", help="Matcher", default="simple")
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
        if type(mat) is list:
            mat = mat[0]

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



def flaanMatcher(img1, img2, kp1, kp2, des1, des2):
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

    good_matches = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)

        # if m.distance >= 0.6 * n.distance:
        #     print "Discarting matching: %s %s" % (m.distance, (0.6 * n.distance))
        #     matchesMask[i] = [1,0]


    # for (int i = 0; i < keypoints_1.size(); ++i)
    # {
    #   if (matches[i].size() < 2)
    #       continue;
    #
    #   const DMatch &m1 = matches[i][0];
    #   const DMatch &m2 = matches[i][1];
    #
    #   if (m1.distance <= 0.6 * m2.distance)
    #       good_matches.push_back(m1);
    # }

    '''
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)
    '''
    drawMatches(img1, kp1, img2, kp2, good_matches)





def simpleMatcher(img1, img2, kp1, kp2, des1, des2):
    # create BFMatcher object
    bf = cv2.BFMatcher()

    # Matches and sort them in the order of their distance.
    matches = bf.match(des1,des2)
    drawMatches(img1, kp1, img2, kp2, matches)


# Get args
templatePath = args['template']
queryPath = args['query']
hessian = float(args['hessian'])

# Load images
imgTemplate = cv2.imread(templatePath, 0)
imgQuery = cv2.imread(queryPath, 0)

# Create SURF
surf = cv2.xfeatures2d.SURF_create(hessian)
surf.setUpright(True)
surf.setHessianThreshold(hessian)

# Detect
kp1, des1 = surf.detectAndCompute(imgTemplate,None)
kp2, des2 = surf.detectAndCompute(imgQuery,None)


argMatcher = args['matcher']

if 'simple' == argMatcher:
    # Using simple matcher
    print 'Matcher: ' + argMatcher
    simpleMatcher(imgTemplate, imgQuery, kp1, kp2, des1, des2)

elif 'flann' == argMatcher:
    # Using flann matcher
    print 'Matcher: ' + argMatcher
    flaanMatcher(imgTemplate, imgQuery, kp1, kp2, des1, des2)

else:
    print 'Unknown matcher: ' + argMatcher

print 'Done !!!'
print ''
