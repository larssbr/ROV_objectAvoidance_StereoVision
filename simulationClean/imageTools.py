import cv2
import numpy as np
import math
import sys

from matplotlib import pyplot as plt


class centroidTools:

    def __init__(self, imgBW):
        # self.MESSAGE = ObstacleAvoidance.getMessage()
        self.imgBW = self.prepareDisparityImage_for_centroid(imgBW) # BW for black and white
        self.centerCordinates = []
        self.objectCenter = (0,0)

        self.focallength_mm = (2222.72426 * 35) / 1360
        self.pxPERmm = 2222.72426 / self.focallength_mm  # pxPERmm = 38.8571428572

        self.pixelSizeOfObject = 50

    def findCentroids(self):    # todo: change name to findCentroidsCenterCords(self)
        imgBWCopy = self.imgBW.astype(np.uint8)

       #h, w = imgBW.shape[:2]
        #contours0, hierarchy = cv2.findContours( imgBW.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours0, hierarchy = cv2.findContours( imgBWCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        moments = [cv2.moments(cnt) for cnt in contours0]

        # rounded the centroids to integer.
        centroids = [( int(round(m['m10']/m['m00'])), int(round(m['m01']/m['m00'])) ) for m in moments]

        #centerCordinates = []

        for ctr in centroids:
            # draw a black little empty circle in the centroid position
            centerCircle_Color = (0, 0, 0)
            cv2.circle(self.imgBW, ctr, 4, centerCircle_Color)
            self.centerCordinates.append(ctr)
            #ctrTimes = 1

        self.centerCordinates = np.asarray(self.centerCordinates)

        return self.imgBW, self.centerCordinates

    def get_centerCordinates(self):
        return self.centerCordinates

    def getAverageCentroidPosition(self):
        # taking the average of the centroids x and y poition to calculate and estimated  object CENTER
        # centerCordinates = centerCordinates.astype(np.int)
        self.objectCenter = np.average(self.centerCordinates, axis=0)
        # objectCenter = objectCenter.astype(np.uint8)

        print "objectCenter  : "
        print self.objectCenter  # , objectCentery
        # print objectCentery

        # Unpack tuple.
        (objectCenterX, objectCenterY) = self.objectCenter

        # Display unpacked variables.
        print(objectCenterX)
        print(objectCenterY)
        # pack the tupple
        self.objectCenter = (int(objectCenterX), int(objectCenterY))

        return self.objectCenter

    def calcDistanceToKnownObject(self, object_real_world_mm):
        object_image_sensor_mm = self.pixelSizeOfObject / self.pxPERmm
        distance_mm = (object_real_world_mm * self.focallength_mm) / object_image_sensor_mm
        return distance_mm

    def prepareDisparityImage_for_centroid(self, IMGbw):
        # IMGbw = cv2.erode(IMGbw, np.ones((4, 4)))

        # DILATE white points...
        IMGbw = cv2.dilate(IMGbw, np.ones((5, 5)))
        IMGbw = cv2.dilate(IMGbw, np.ones((5, 5)))
        # IMGbw = cv2.erode(IMGbw, np.ones((4, 4)))
        # IMGbw = cv2.dilate(IMGbw, np.ones((5, 5)))
        return IMGbw

    # object avoidance      #################
    def findBiggestObject(self, imgBW, pts_que_center, pts_que_radius, radiusTresh=40):


        blurred = cv2.GaussianBlur(imgBW, (11, 11), 0)
        mask = blurred
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        biggestObjectCenter = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            biggestObjectCenter = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > radiusTresh:  # works as a treshold
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(imgBW, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(imgBW, biggestObjectCenter, 5, (0, 0, 255), -1)

        # update the points queue
        try:
            pts_que_center.appendleft(biggestObjectCenter)
            pts_que_radius.appendleft(radius)
            pts_que_center_List = list(pts_que_center)
            pts_que_radius_List = list(pts_que_radius)
        except:
            pass

        # loop over the set of tracked points
        for i in xrange(1, len(pts_que_center)):
            # if either of the tracked points are None, ignore
            # them
            if pts_que_center[i - 1] is None or pts_que_center[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(15 / float(i + 1)) * 2.5)
            cv2.line(imgBW, pts_que_center[i - 1], pts_que_center[i], (255, 255, 255), thickness)

        return imgBW, biggestObjectCenter, pts_que_center_List, pts_que_radius_List


class messageTools:
    def __init__(self, MESSAGE, objectCenter):
        self.MESSAGE = MESSAGE
        self.objectCenter = objectCenter
        #self.MESSAGE = ObstacleAvoidance.getMessage()

    def findXposMessage(self):
        cx, cy = self.objectCenter
        # make new "coordinate system"
        middleX = 1360 / 2  # 680

        if cx < middleX:
            Xpos = - (middleX - cx)  # - 50 is a little to the left
        else:
            Xpos = cx - middleX

        return -Xpos  # - to send the direction the rov should go, more intutive then where it should not go

    def findYposMessage(self):
        cx, cy = self.objectCenter
        # make new "coordinate system"
        middleY = 1024 / 2  # 512

        if cy < middleY:
            Ypos = - (middleY - cx)  # - 50 is a little to the left
        else:
            Ypos = cx - middleY

        return -Ypos  # - to send the direction the rov should go, more intutive then where it should not go


class drawTools:
    def __init__(self, image, centerCordinates, objectAVGCenter):
        #self.Xpath = Xpath
        #self.ypos = ypos
        self.image = image
        self.centerCordinates = centerCordinates
        self.objectAVGCenter = objectAVGCenter


    def drawPath(self, Xpath, Ypos, image):
        radius = 100
        # (x,y) = centerCordinates

        # center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(image, (Xpath, Ypos), radius, (255, 255, 255), 7)

        return image

    def drawBox(self):
        ############## creating a minimum rectangle around the object ######################
        try:
            rect = cv2.minAreaRect(points=self.centerCordinates)
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(self.image, [box], 0, (255, 255, 255), 2)
        except:
            pass

    def circle_around_object(self):
        ########### circle around object #######3

        try:
            (x, y), radius = cv2.minEnclosingCircle(self.centerCordinates)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(self.image, center, radius, (255, 255, 255), 2)
        except:
            pass

    def elipse_around_object(self):
        ########### finding a elipse ##############
        # if len(centerCordinates) > 5:  # need more than points than 5 to be able to run  cv2.fitEllipse
        try:
            ellipse = cv2.fitEllipse(self.centerCordinates)
            cv2.ellipse(self.image, ellipse, (255, 255, 255), 2)
        except:
            pass

    def fitting_line_thruObject(self):
        ##### fitting a line ###########
        try:
            rows, cols = self.image.shape[:2]
            [vx, vy, x, y] = cv2.fitLine(points=self.centerCordinates, distType=cv2.cv.CV_DIST_L2, param=0, reps=0.01,
                                         aeps=0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)
            cv2.line(self.image, (cols - 1, righty), (0, lefty), (255, 255, 255), 2)
        except:
            pass

    # drawTextMessage(image_color_with_Draw, str(distance_mm))
    def drawTextMessage(self, text):
        try:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.image, text, (10, 500), font, 1, (255, 255, 255), 2)

        except:
            pass

    def drawAVGcenter_circle(self):
        # draw the new objectAVGCenter in a white circle
        centerCircle_Color = (255, 255, 255)
        cv2.circle(self.image, self.objectAVGCenter, 10, centerCircle_Color)

    def get_drawnImage(self):
        return self.image


    def saveImage(self, image_name_str, image):
        cv2.imwrite(image_name_str, image)

    def drawStuff(self, centerCordinates):
        # http://opencvpython.blogspot.no/2012/06/contours-2-brotherhood.html
        # http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html#gsc.tab=0pyth
        ############## creating a minimum rectangle around the object ######################
        try:
            rect = cv2.minAreaRect(points=centerCordinates)
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(self.image, [box], 0, (255, 255, 255), 2)
        except:
            pass
        ########### circle around object #######3

        try:
            (x, y), radius = cv2.minEnclosingCircle(centerCordinates)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(self.image, center, radius, (255, 255, 255), 2)
        except:
            pass

        ########### finding a elipse ##############
        # if len(centerCordinates) > 5:  # need more than points than 5 to be able to run  cv2.fitEllipse
        try:
            ellipse = cv2.fitEllipse(centerCordinates)
            cv2.ellipse(self.image, ellipse, (255, 255, 255), 2)
        except:
            pass

        ##### fitting a line ###########

        try:
            rows, cols = self.image.shape[:2]
            [vx, vy, x, y] = cv2.fitLine(points=centerCordinates, distType=cv2.cv.CV_DIST_L2, param=0, reps=0.01,
                                         aeps=0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)
            cv2.line(self.image, (cols - 1, righty), (0, lefty), (255, 255, 255), 2)
        except:
            pass

        try:
            pixelSizeOfObject = radius  # an okay estimate for testing
        except:
            pixelSizeOfObject = 50

        return self.image, pixelSizeOfObject


##############################################################################################################################################################
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
    (T, thresh) = cv2.threshold(img.astype(np.uint8), 100, 255, cv2.THRESH_BINARY)

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

