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
ap.add_argument("-m", "--matcher", help="Matcher", default="simple")
ap.add_argument("-v", "--video", help = "path to the (optional) video file")
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
    #cv2.imwrite(args['output'], out)
    if len(matches) > 27:
        cv2.putText(out, "matcher: %s" % len(matches), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0))
    cv2.imshow('matcher', out)



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
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

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

def execute_matcher(imgQuery):
    # Load images
    imgTemplate = cv2.imread(templatePath, 0)

    # Create SURF
    surf = cv2.SURF(hessian)
    surf.upright = True
    surf.hessianThreshold = hessian

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


# if a video path was not supplied, grab the reference
# to the gray
if args.get("video", False):
    camera = cv2.VideoCapture(0)

    #print "fps: %s" % camera.get(cv2.cv.CV_CAP_PROP_FPS)
    camera.set(cv2.cv.CV_CAP_PROP_FPS, 30)

    while True:

        (grabbed, frame) = camera.read()

        #frame = imutils.resize(frame, width = 400)
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        execute_matcher(converted)
        #cv2.imshow('matcher', converted)

        # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

else:
    imgQuery = cv2.imread(queryPath, 0)
    execute_matcher(imgQuery)
    cv2.waitKey(0)

print 'Done !!!'
print ''
