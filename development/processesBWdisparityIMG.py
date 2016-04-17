import cv2
import numpy as np
import math
import sys


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

    cv2.imshow("image after treshold", imgBW )
    cv2.waitKey(0)

    h, w = imgBW.shape[:2]
    contours0, hierarchy = cv2.findContours( imgBW.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #print 'contours0:', contours0
    #print 'hierarchy:', hierarchy
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

        print "ctr"
        print ctr
        print(centerCordinates[0])

    return imgBW, centerCordinates

def getAverageCentroidPosition(centerCordinates):
    # taking the average of the centroids x and y poition to calculate and estimated  object CENTER
    objectCenter = np.average(centerCordinates, axis=0)

    print "objectCenter  : "
    print objectCenter #, objectCentery
    #print objectCentery

    # Unpack tuple.
    (objectCenterX, objectCenterY) = objectCenter

    # Display unpacked variables.
    print(objectCenterX)
    print(objectCenterY)

    objectCenter = (int(objectCenterX), int(objectCenterY))

    '''
    print "objectY  : "
    print  objecty

    #ctrCenter = int(objectX), int(objecty)
    ctrCenter = int(objecty), int(objectX)


    objectX_max = np.ndarray.max(centerX)
    objectX_min = np.ndarray.min(centerX)

    objecty_min = np.ndarray.min(centerY)

    objecty_max = np.ndarray.max(centerY)



    print "objecty_min  : "
    print  objecty_min
    print "objecty_max  : "
    print  objecty_max

    print "objectX_max  : "
    print objectX_max
    print "objectX_min  : "
    print objectX_min
    '''


    return objectCenter


def drawStuff(centerCordinates, image):
    # http://opencvpython.blogspot.no/2012/06/contours-2-brotherhood.html
    # http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html#gsc.tab=0pyth
   ############## creating a minimum rectangle around the object ######################
    rect = cv2.minAreaRect(points=centerCordinates)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image,[box],0,(255,255,255),2)

    ########### circle around object #######3

    (x, y),radius = cv2.minEnclosingCircle(centerCordinates)
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(image, center, radius, (255,255,255),2)

    ########### finding a elipse ##############

    ellipse = cv2.fitEllipse(centerCordinates)
    cv2.ellipse(image,ellipse,(255,255,255),2)

    ##### fitting a line ###########

    rows,cols = image.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(points=centerCordinates, distType=cv2.cv.CV_DIST_L2, param =0, reps=0.01, aeps=0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    cv2.line(image,(cols-1,righty),(0,lefty),(255,255,255),2)

    pixelSizeOfObject = radius  # an okay estimate for testing
    return image, pixelSizeOfObject

def calcDistanceToKnownObject(object_real_world_mm, pixelSizeOfObject):

    focallength_mm = (2222.72426*35)/1360
    pxPERmm = 2222.72426/focallength_mm # pxPERmm = 38.8571428572

    object_image_sensor_mm =pixelSizeOfObject/ pxPERmm
    distance_mm = (object_real_world_mm * focallength_mm) / object_image_sensor_mm
    return distance_mm


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
    cv2.imshow("Threshold Binary", thresh)

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


def mainProcess():

    filename = r"savedImages\tokt1_Depth_map_1.jpg"
    IMGbw = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)


    # Erode to remove noise
    IMGbw = cv2.erode(IMGbw, np.ones((4, 4)))

    # DILATE white points...
    IMGbw = cv2.dilate(IMGbw, np.ones((5, 5)))
    IMGbw = cv2.erode(IMGbw, np.ones((4, 4)))
    IMGbw = cv2.dilate(IMGbw, np.ones((5, 5)))

    # calculate the centers of the small "objects"
    image, centerCordinates = findCentroids(IMGbw)

    cv2.imshow("image after finding centroids", image )
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


    # CALCULATE the distance to this object



if __name__ == '__main__':
    mainProcess()