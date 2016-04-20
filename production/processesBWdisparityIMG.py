import cv2
import numpy as np
import math
import sys

from matplotlib import pyplot as plt


##### helper methods

##
# Converts an image into a binary image at the specified threshold.
# All pixels with a value <= threshold become 0 --> BLACK, while
# pixels > threshold become 1 --> WHITE
def use_threshold(image, threshold = 100):
    (thresh, im_bw) = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return (thresh, im_bw)

##
# Finds the outer contours of a binary image and returns a shape-approximation
# of them. Because we are only finding the outer contours, there is no object
# hierarchy returned.
##
def find_contours(image):
    (contours, hierarchy) = cv2.findContours(image, mode=cv2.cv.CV_RETR_EXTERNAL, method=cv2.cv.CV_CHAIN_APPROX_SIMPLE)
    return contours

##
# Finds the centroids of a list of contours returned by
# the find_contours (or cv2.findContours) function.
# If any moment of the contour is 0, the centroid is not computed. Therefore
# the number of centroids returned by this function may be smaller than
# the number of contours passed in.
#
# The return value from this function is a list of (x,y) pairs, where each
# (x,y) pair denotes the center of a contour.
##
def find_centers(contours):
    centers = []
    for contour in contours:
        moments = cv2.moments(contour, True)

        # If any moment is 0, discard the entire contour. This is
        # to prevent division by zero.
        if (len(filter(lambda x: x==0, moments.values())) > 0):
            continue

        center = (moments['m10']/moments['m00'] , moments['m01']/moments['m00'])
        # Convert floating point contour center into an integer so that
        # we can display it later.

        center = map(lambda x: int(round(x)), center)
        centers.append(center)
    return centers

##
# Draws circles on an image from a list of (x,y) tuples
# (like those returned from find_centers()). Circles are
# drawn with a radius of 20 px and a line width of 2 px.
##
def draw_centers(centers, image):
    for center in centers:
        cv2.circle(image, tuple(center), 20, cv2.cv.RGB(0,255,255), 2)

def findCentroids(imgBW):
    #_,imgBW = cv2.threshold(imgBW, 0, 255, cv2.THRESH_OTSU)
    #(thresh, imgBW) = use_threshold(imgBW, threshold = 10)

    #cv2.imshow("image after treshold", imgBW )

    imgBWCopy = imgBW.astype(np.uint8)

    h, w = imgBW.shape[:2]
    #contours0, hierarchy = cv2.findContours( imgBW.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours0, hierarchy = cv2.findContours( imgBWCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    moments = [cv2.moments(cnt) for cnt in contours0]

    # rounded the centroids to integer.
    centroids = [( int(round(m['m10']/m['m00'])), int(round(m['m01']/m['m00'])) ) for m in moments]

    centerCordinates = []

    #print 'centroids:', centroids
    # TODO: only get the biggest centroid....
    for ctr in centroids:
        # draw a black little empty circle in the centroid position
        centerCircle_Color = (0, 0, 0)
        cv2.circle(imgBW, ctr, 4, centerCircle_Color)

        centerCordinates.append(ctr)
        #cv2.circle(imgBW, ctr, 5, (0,0,0))
        ctrTimes = 1
        #if ctrTimes == 1:  # Only wants the first ctr
        #    break

        #print "ctr"
        #print ctr
        #print(centerCordinates[0])

    return imgBW, centerCordinates

def getAverageCentroidPosition(centerCordinates):
    # taking the average of the centroids x and y poition to calculate and estimated  object CENTER
    #centerCordinates = centerCordinates.astype(np.int)
    objectCenter = np.average(centerCordinates, axis=0)
    #objectCenter = objectCenter.astype(np.uint8)

    print "objectCenter  : "
    print objectCenter #, objectCentery
    #print objectCentery

    # Unpack tuple.
    (objectCenterX, objectCenterY) = objectCenter

    # Display unpacked variables.
    print(objectCenterX)
    print(objectCenterY)

    objectCenter = (int(objectCenterX), int(objectCenterY))

    return objectCenter


def drawPath(Xpath,Ypos, image):
    radius = 100
    #(x,y) = centerCordinates

    #center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(image, (Xpath,Ypos), radius, (255,255,255), 7)



    return image

def drawStuff(centerCordinates, image):
    # http://opencvpython.blogspot.no/2012/06/contours-2-brotherhood.html
    # http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html#gsc.tab=0pyth
   ############## creating a minimum rectangle around the object ######################
    try:
        rect = cv2.minAreaRect(points=centerCordinates)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image,[box],0,(255,255,255),2)
    except:
        pass
    ########### circle around object #######3

    try:
        (x, y),radius = cv2.minEnclosingCircle(centerCordinates)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(image, center, radius, (255,255,255),2)
    except:
        pass


    ########### finding a elipse ##############
    #if len(centerCordinates) > 5:  # need more than points than 5 to be able to run  cv2.fitEllipse
    try:
        ellipse = cv2.fitEllipse(centerCordinates)
        cv2.ellipse(image,ellipse,(255,255,255),2)
    except:
        pass

    ##### fitting a line ###########

    try:
        rows,cols = image.shape[:2]
        [vx,vy,x,y] = cv2.fitLine(points=centerCordinates, distType=cv2.cv.CV_DIST_L2, param =0, reps=0.01, aeps=0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        cv2.line(image,(cols-1,righty),(0, lefty),(255,255,255),2)
    except:
        pass

    try:
        pixelSizeOfObject = radius  # an okay estimate for testing
    except:
        pixelSizeOfObject = 50

    return image, pixelSizeOfObject

def calcDistanceToKnownObject(object_real_world_mm, pixelSizeOfObject):

    focallength_mm = (2222.72426*35)/1360
    pxPERmm = 2222.72426/focallength_mm # pxPERmm = 38.8571428572

    object_image_sensor_mm =pixelSizeOfObject/ pxPERmm
    distance_mm = (object_real_world_mm * focallength_mm) / object_image_sensor_mm
    return distance_mm

def findXposMessage(objectCenter):
    cx, cy = objectCenter
    # make new "coordinate system"
    middleX = 1360/2 # 680

    if cx < middleX :
        Xpos = - (middleX - cx) # - 50 is a little to the left
    else:
        Xpos = cx - middleX

    return -Xpos   # - to send the direction the rov should go, more intutive then where it should not go

def findYposMessage(objectCenter):
    cx, cy = objectCenter
    # make new "coordinate system"
    middleY = 1024/2 # 512

    if cy < middleY :
        Ypos = - (middleY - cx) # - 50 is a little to the left
    else:
        Ypos = cx - middleY

    return -Ypos   # - to send the direction the rov should go, more intutive then where it should not go

# Not used
def findCentroifOfObject(img):
    '''

    Accepts an BW image as Numpy array
       Returns: (x,y) coordinates of centroid if found
                (-1,-1) if no centroid was found
                None if user hit ESC
    '''
    # grayBoundary = ([103, 86, 65], [145, 133, 128])
    lower = np.array([103, 86, 65])
    upper = np.array([145, 133, 128])
    ######### Treshold #############3

    #blurred = cv2.GaussianBlur(img, (5, 5), 0)
    #cv2.imshow("Image with gaussianBlur", img)

    #(T, thresh) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)
    (T, thresh) = cv2.threshold(img.astype(np.uint8), 100, 255, cv2.THRESH_BINARY)
    #cv2.imshow("Threshold Binary", thresh)

    #(T, threshInv) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV)
    #(T, threshInv) = cv2.threshold(img.astype(np.uint8), 100, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("Threshold Binary Inverse", threshInv)

    #coins = cv2.bitwise_and(img, img, mask =threshInv)
    #cv2.imshow("Coins", coins)
    #cv2.waitKey(0)

    # find the colors within the specified boundaries and apply
    # the mask
    #mask = cv2.inRange(img, lower, upper)
    #output = cv2.bitwise_and(img, img, mask = mask)

    # Blur the mask
    #bmask = cv2.GaussianBlur(mask, (5,5),0)

    # Take the moments to get the centroid
    moments = cv2.moments(thresh)  #bmask)
    m00 = moments['m00']
    print m00
    centroid_x, centroid_y = None, None
    if m00 != 0:
        print "m00 != 0"
        centroid_x = int(moments['m10']/m00)
        centroid_y = int(moments['m01']/m00)

    # Assume no centroid
    ctr = (-1,-1)

    # Use centroid if it exists
    if centroid_x != None and centroid_y != None:

        ctr = (centroid_x, centroid_y)

        # Put black circle in at centroid in image
        centerCircle_Color = (62, 174, 250)
        cv2.circle(img, ctr, 4, centerCircle_Color)

    # Display full-color image
    WINDOW_NAME = "ObjectFinder"
    #cv2.imshow(WINDOW_NAME, img)
    #cv2.waitKey(0)

    # Force image display, setting centroid to None on ESC key input
    if cv2.waitKey(1) & 0xFF == 27:
        ctr = None

    # want to return position of centroid in picture

    return img, ctr # we return the image with the centroid center -->ctr = (centroid_x, centroid_y)
    # want to calculate average distance to this centroid as well


    #cv2.destroyAllWindows()

def objectTreshold(leftImg, rightImage):
    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(leftImg, None)
    kp2, des2 = sift.detectAndCompute(rightImage, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1 ,des2, k=2)

    acceptedINT = 0
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.3*n.distance:
            good.append([m])
            acceptedINT = acceptedINT + 1

    matchesINT = len(matches)
    return matchesINT #acceptedINT

def prepareDisparityImage_for_centroid(IMGbw):
    #IMGbw = cv2.erode(IMGbw, np.ones((4, 4)))

    # DILATE white points...
    IMGbw = cv2.dilate(IMGbw, np.ones((5, 5)))
    IMGbw = cv2.dilate(IMGbw, np.ones((5, 5)))
    #IMGbw = cv2.erode(IMGbw, np.ones((4, 4)))
    #IMGbw = cv2.dilate(IMGbw, np.ones((5, 5)))
    return IMGbw

def mainProcess():

    filename = r"savedImages\tokt1_Depth_map_1.jpg"
    IMGbw = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    #filenameLeft = r"savedImages\tokt1_L_1.jpg"
    #filenameRight = r"savedImages\tokt1_R_1.jpg"

    filenameLeft = r"testImages\lawnMowerPattern_seafloor\151210-134618.3229L.jpg"
    filenameRight = r"testImages\lawnMowerPattern_seafloor\151210-134618.3229R.jpg"

    filenameLeft = r"testImages\lawnMowerPattern_seafloor\151210-134628.3236L.jpg"
    filenameRight = r"testImages\lawnMowerPattern_seafloor\151210-134628.3236R.jpg"

    filenameLeft = r"testImages\obstacle1\RightCameraRun4_179.png"
    filenameRight = r"testImages\obstacle1\LeftCameraRun4_179.png"

    IMG_L = cv2.imread(filenameLeft)
    IMG_R = cv2.imread(filenameRight)

    # Erode to remove noise
    IMGbw_PrepcalcCentroid = prepareDisparityImage_for_centroid(IMGbw)

    # calculate the centers of the small "objects"
    image, centerCordinates = findCentroids(IMGbw_PrepcalcCentroid)
    cv2.imshow("image after finding centroids", image)
    cv2.waitKey(0)

    centerCordinates = np.asarray(centerCordinates) # make list centerCordinates into numpy array

    imageDraw, pixelSizeOfObject = drawStuff(centerCordinates, image.copy())

    object_real_world_mm = 500 # 1000mm = 1 meter
    distance_mm = calcDistanceToKnownObject(object_real_world_mm, pixelSizeOfObject)
    # calculate an estimate of distance

    print "distance_mm"
    print distance_mm

    cv2.imshow("image after finding minimum bounding rectangle of object", imageDraw )
    cv2.waitKey(0)

    objectCenter = getAverageCentroidPosition(centerCordinates)

    #draw the new center in white
    centerCircle_Color = (255, 255, 255)
    #centerCircle_Color = (0, 0, 0)
    cv2.circle(image, objectCenter, 10, centerCircle_Color)

    cv2.imshow('center object', image)
    cv2.waitKey(0)

    # get direction
    Xpos = findXposMessage(objectCenter)
    print "Xpos"
    print Xpos



    # check if there is a object in the image by using the treshold algorithm
    #matchINT = objectTreshold(IMG_L, IMG_R)
    #print "matchINT"
   # print matchINT

    # CALCULATE the distance to this object using a point cloud


if __name__ == '__main__':
    mainProcess()