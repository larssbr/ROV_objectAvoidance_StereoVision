

from pymba import *
import numpy as np
import cv2
import time
import cProfile
import math # to get math.pi value
import socket # to send UDP message to labview pc
import disparityMapCalc as disp
import processesBWdisparityIMG as proc
from collections import deque # to keep track of a que of last poitions of the center of an object
import datetime
import drawing as draw


#import stereo as stereo

def printTime():
    time.time()

def dispalyToUser(img1,img2):
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)

def disparityCalc(img1, img2, intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR):
    ############# CALCULATE Disparity ############################
    #print('Undistort the left images')
    undistorted_image_L = disp.UndistortImage(img1, intrinsic_matrixL, distCoeffL)

    # Secondly undistort images taken with right camera
    #print('Undistort the right images')
    undistorted_image_R = disp.UndistortImage(img2, intrinsic_matrixR, distCoeffR)


    #cv2.imshow("undistorted_image_L", undistorted_image_L)
    #cv2.imshow("undistorted_image_R", undistorted_image_R)
    #cv2.waitKey(0)

    # --> calculate disparity images
    disparity_visual = disp.getDisparity(imgLeft=undistorted_image_L, imgRight=undistorted_image_R, method="BM")
    disparity_visual = disparity_visual.astype(np.uint8)
    return disparity_visual

def disparityDisctance(disparity_visual, focal_length, base_offset):
    #D:= Distance of point in real world,
    #b:= base offset, (the distance *between* your cameras)
    #f:= focal length of camera,
    #d:= disparity:

    #D = b*f/d
    Depth_map = (base_offset*focal_length)/disparity_visual
    return Depth_map

### Helper methods ###
def getImg(frame_data, frame):
    img = np.ndarray(buffer=frame_data,
                     dtype=np.uint8,
                     #shape=(frame.height,frame.width,1))
                     shape=(frame.height,frame.width,frame.pixel_bytes))

    return img

# comunication methods

def sendUDPmessage(MESSAGE):

    # addressing information of target
    UDP_IP = "192.168.1.100"
    UDP_PORT = 1130
    #MESSAGE = "Hello, World!"

    #print "UDP target IP:", UDP_IP
    #print "UDP target port:", UDP_PORT
    print "message:", MESSAGE

    # initialize a socket, think of it as a cable
    # SOCK_DGRAM specifies that this is UDP
    try:
        sock = socket.socket(socket.AF_INET, # Internet
                             socket.SOCK_DGRAM) # UDP
         # send the command
        sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))

        # close the socket
        sock.close()
    except:
        pass

#################################################

def percentageBlack(imgROI):
    width, height = imgROI.shape[:2][::-1]
    totalNumberOfPixels = width * height
    ZeroPixels = totalNumberOfPixels - cv2.countNonZero(imgROI)
    return ZeroPixels

def meanPixelSUM(imgROI):
    '''
    sum = np.sum(imgROI)  #cv2.sumElems(imgROI)
    width, height = imgROI.shape[:2][::-1]
    totalNumberOfPixels = width * height
    mean = sum/totalNumberOfPixels
    '''
    meanValue = cv2.mean(imgROI)

    return meanValue[0] # want to have the 0 array, else the data is represented liek this: (1.3585184021230532, 0.0, 0.0, 0.0)

def obstacleAvoidanceDirection(img):

    width, height = img.shape[:2][::-1]
    margin_sides = 100
    width_parts = (width-margin_sides)/3
    hight_margin = height/15

    Left_piece = img[0:height, 0:453]
    Center_piece = img[0:height, 453:906]
    Right_piece = img[0:height, 906:1360]

    # trying with margins
    # A predefined frame is subtracted in order to avoid mismatches due to different field of view of the ztwo images.
    Left_piece = img[hight_margin:height-hight_margin, margin_sides/3:width_parts]
    Center_piece = img[hight_margin:height- hight_margin, width_parts:width_parts*2]
    Right_piece = img[hight_margin:height-hight_margin, width_parts*2:width_parts*3]


    # Which of the areas has the least amount of "obstacles"
    #Left_piece_INT = percentageBlack(Left_piece)
    #Center_piece_INT = percentageBlack(Center_piece)
    #Right_piece_INT = percentageBlack(Right_piece)

    Left_piece_INT = meanPixelSUM(Left_piece)
    Center_piece_INT = meanPixelSUM(Center_piece)
    Right_piece_INT = meanPixelSUM(Right_piece)

    print Left_piece_INT
    print Center_piece_INT
    print Right_piece_INT

    if Left_piece_INT > Center_piece_INT and Left_piece_INT > Right_piece_INT:
        return "LEFT"
    if Center_piece_INT > Left_piece_INT and Center_piece_INT > Right_piece_INT:
        return "CENTER"
    if Right_piece_INT > Center_piece_INT and Right_piece_INT > Left_piece_INT:
        return "RIGHT"
    else:
        return "NEED MORE TIME TO DECIDE"

def isObsticleInFront(img, isObsticleInFrontTreshValue):
    meanValue = meanPixelSUM(img)
    print "meanValue for disparity image"
    print meanValue
    if meanValue < isObsticleInFrontTreshValue: #1.7:
        return True
    else:
        return False

#######################################################

def camereaAngleAdjuster(img):
    # Imporve disparity image, by using a scale sin(20) to sin(50) --> becouse the camera is tilted 35 or 45 degrees?
    # make an array of values from sin(20) to sin(50)
    width, height = img.shape[:2][::-1]

    steps = height/(50-20)
    # sin(20) = 0.91294525072 (rad) = 0.5 (deg)
    # sin(50) -0.2623748537 (rad) = 0.76604444311 (deg)
    startRad = deg2rad(20)
    stopRad = deg2rad(50)

    divideBy = deg2rad(35)
    #/sin(35)

    #angleScalarList = np.linspace(np.sin(startRad),np.sin(stopRad), height) # accending order

    angleScalarList = np.linspace(np.sin(stopRad)/np.sin(divideBy), np.sin(startRad)/np.sin(divideBy), height) # decreasing order

     # then multiply that array by the matrix
    for i,value  in enumerate(angleScalarList):
        img[i,:] = img[i,:]*value

    return img

# HELPER FUNCTIONS degrees and radians
def rad2deg(radians):
    # degrees = 180 * radians / pi
    pi = math.pi
    degrees = 180 * radians / pi
    return degrees

def deg2rad(degrees):
    # radians = pi * degrees / 180
    pi = math.pi
    radians = pi * degrees / 180
    return radians

##################Point Cloud CODE################################

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def point_cloud(disparity_image, image_left, focal_length):
    """Create a point cloud from a disparity image and a focal length.

    Arguments:
      disparity_image: disparities in pixels.
      image_left: BGR-format left stereo image, to color the points.
      focal_length: the focal length of the stereo camera, in pixels.

    Returns:
      A string containing a PLY point cloud of the 3D locations of the
        pixels, with colors sampled from left_image. You may filter low-
        disparity pixels or noise pixels if you choose.
    """

    h, w = image_left.shape[:2]
    Q = np.float32([[1, 0, 0, w / 2],
                    [0, -1, 0, h / 2],
                    [0, 0, focal_length, 0],
                    [0, 0, 0, 1]])
    points = cv2.reprojectImageTo3D(disparity_image, Q)
    colors = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
    mask = disparity_image > disparity_image.min()
    out_points = points[mask]
    out_colors = colors[mask]

    verts = out_points.reshape(-1, 3)
    colors = out_colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])

    ply_string = ply_header % dict(vert_num=len(verts))
    for vert in verts:
        ply_string += "%f " % vert[0]
        ply_string += "%f " % vert[1]
        ply_string += "%f " % vert[2]
        ply_string += "%d " % vert[3]
        ply_string += "%d " % vert[4]
        ply_string += "%d\n" % vert[5]

    print "saving pointCloud"
    with open("set.ply", 'w') as f:
        f.write(ply_string)

    #return result # ply_string

def findBiggestObject(img, pts_que_center, radiusTresh, isObstacleInfront_based_on_radius):
    width, height = img.shape[:2][::-1]

    margin = 200
    #img= img[margin:width-margin , 0 : height]
    y1 = 0
    y2 = height
    x1 = margin
    x2 = width-margin
    img = img[y1:y2, x1:x2]

    blurred = cv2.GaussianBlur(img, (11, 11), 0)
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
        print "radius is ##################"
        print radius
        M = cv2.moments(c)
        biggestObjectCenter = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > radiusTresh: # works as a treshold
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(img, (int(x), int(y)), int(radius), (255, 255, 255), 2)
            cv2.circle(img, biggestObjectCenter, 5, (255, 255, 255), -1)
            # set isObstacleInfront_based_on_radius to True
            isObstacleInfront_based_on_radius = True
        else:
            isObstacleInfront_based_on_radius = False

    # update the points queue
    pts_que_center.appendleft(biggestObjectCenter)
    #pts_que_radius.appendleft(radius)
    pts_que_center_List = list(pts_que_center)
    #pts_que_radius_List = list(pts_que_radius)


    # loop over the set of tracked points
    for i in xrange(1, len(pts_que_center)):
        # if either of the tracked points are None, ignore
        # them
        if pts_que_center[i - 1] is None or pts_que_center[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(15/ float(i + 1)) * 2.5)
        cv2.line(img, pts_que_center[i - 1], pts_que_center[i], (255, 255, 255), thickness)

    return img, biggestObjectCenter, pts_que_center_List, isObstacleInfront_based_on_radius


def trackPathPos(pts_que_path_center, xpos, ypos):
    objectCenter = xpos, ypos
    pts_que_path_center.appendleft(objectCenter)
    pts_que_path_center_List = list(pts_que_path_center)

    return pts_que_path_center_List

def drawTrackedPoints(img,pts_que_center):
    # loop over the set of tracked points
    for i in xrange(1, len(pts_que_center)):
        # if either of the tracked points are None, ignore
        # them
        if pts_que_center[i - 1] is None or pts_que_center[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(15/ float(i + 1)) * 2.5)
        cv2.line(img, pts_que_center[i - 1], pts_que_center[i], (255, 255, 255), thickness)

    return img


def trackGrowingObject(img, pts_que_center_list, pts_que_radius_list, radiusTresh=20):
    # pts_que_center_list
    #pts_que_center = deque(maxlen=15)

    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    mask = blurred
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
	# (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the 5 largest contours in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        c_list = sorted(cnts,cmp=None,key=cv2.contourArea, reverse=True)
        maxNumber = len(pts_que_center_list)
        for j in xrange(1, len(c_list)):
            if j < maxNumber:
                #pts_que_center_list[j] = c_list[j]
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c_list[j])
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                pts_que_center_list[j].appendleft(center)
                pts_que_radius_list[j].appendleft(radius)

                # only proceed if the radius meets a minimum size
                if radius > radiusTresh: # works as a treshold
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(img, (int(x), int(y)), int(radius),
                        (0, 255, 255), 2)
                    cv2.circle(img, center, 5, (255, 255, 255), -1)



        #((x, y), radius) = cv2.minEnclosingCircle(c)
        #M = cv2.moments(c)
        #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))



    # update the points queue
    #pts_que_center_list[0].appendleft(center)

    # loop over the set of tracked points
    for j in xrange(1, len(pts_que_center_list)):
        for i in xrange(1, len(pts_que_center_list[j])):
            # if either of the tracked points are None, ignore
            # them
            if pts_que_center_list[j][i - 1] is None or pts_que_center_list[j][i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(15/ float(i + 1)) * 2.5)
            cv2.line(img, pts_que_center_list[j][i - 1], pts_que_center_list[j][i], (255, 255, 255), thickness)

    return img, pts_que_center_list, pts_que_radius_list

def CheckHorizontalInvariance(leftImage, rightImage, laplacian_left, laplacian_right, j, i, state):
    '''
    :param leftImage:
    :param rightImage:
    :param laplacian_left:
    :param laplacian_right:
    :param j:
    :param i:
    :param state:
    :return:
    '''
    sobelR= [1]
    pxX = 1
    pxY = 1
    PushbroomStereoState = False

    #// init parameters
    blockSize = state.blockSize
    disparity = state.zero_dist_disparity
    sobelLimit = state.sobelLimit
    '''
        Mat sobelR, int pxX, int pxY, PushbroomStereoState state) {

    // init parameters
    int blockSize = state.blockSize;
    int disparity = state.zero_dist_disparity;
    int sobelLimit = state.sobelLimit;

    // top left corner of the SAD box
    int startX = pxX;
    int startY = pxY;

    // bottom right corner of the SAD box
    int endX = pxX + blockSize - 1;
    int endY = pxY + blockSize - 1;


    // if we are off the edge of the image so we can't tell if this
    // might be an issue -- give up and return true
    // (note: this used to be false and caused bad detections on real flight
    // data near the edge of the frame)
    if (   startX + disparity + INVARIANCE_CHECK_HORZ_OFFSET_MIN < 0
        || endX + disparity + INVARIANCE_CHECK_HORZ_OFFSET_MAX > rightImage.cols) {

        return true;
    }

    if (startY + INVARIANCE_CHECK_VERT_OFFSET_MIN < 0
        || endY + INVARIANCE_CHECK_VERT_OFFSET_MAX > rightImage.rows) {
        // we are limited in the vertical range we can check here

        // TODO: be smarter here

        // give up and bail out, deleting potential hits
        return true;

    }


    // here we check a few spots:
    //  1) the expected match at zero-disparity (10-infinity meters away)
    //  2) inf distance, moved up 1-2 pixels
    //  3) inf distance, moved down 1-2 pixels
    //  4) others?

    // first check zero-disparity
    int leftVal = 0;

    int right_val_array[400];
    int sad_array[400];
    int sobel_array[400];

    for (int i=0;i<400;i++) {
        right_val_array[i] = 0;
        sad_array[i] = 0;
        sobel_array[i] = 0;
    }

    int counter = 0;

    for (int i=startY;i<=endY;i++)
    {
        for (int j=startX;j<=endX;j++)
        {
            // we are now looking at a single pixel value
            uchar pxL = leftImage.at<uchar>(i,j);

            uchar pxR_array[400], sR_array[400];

            // for each pixel in the left image, we are going to search a bunch
            // of pixels in the right image.  We do it this way to save the computation
            // of dealing with the same left-image pixel over and over again.

            // counter indexes which location we're looking at for this run, so for each
            // pixel in the left image, we examine a bunch of pixels in the right image
            // and add up their results into different slots in sad_array over the loop
            counter = 0;

            for (int vert_offset = INVARIANCE_CHECK_VERT_OFFSET_MIN;
                vert_offset <= INVARIANCE_CHECK_VERT_OFFSET_MAX;
                vert_offset+= INVARIANCE_CHECK_VERT_OFFSET_INCREMENT) {

                for (int horz_offset = INVARIANCE_CHECK_HORZ_OFFSET_MIN;
                    horz_offset <= INVARIANCE_CHECK_HORZ_OFFSET_MAX;
                    horz_offset++) {

                    pxR_array[counter] = rightImage.at<uchar>(i + vert_offset, j + disparity + horz_offset);
                    sR_array[counter] = sobelR.at<uchar>(i + vert_offset, j + disparity + horz_offset);
                    right_val_array[counter] += sR_array[counter];

                    sad_array[counter] += abs(pxL - pxR_array[counter]);


                    counter ++;
                }
            }

            uchar sL = sobelL.at<uchar>(i,j);

            leftVal += sL;

        }
    }

    for (int i = 0; i < counter; i++)
    {
        sobel_array[i] = leftVal + right_val_array[i];

        // we don't check for leftVal >= sobelLimit because we have already
        // checked that in the main search loop (in GetSAD).
        //if (right_val_array[i] >= sobelLimit && 100*(float)sad_array[i]/(float)((float)sobel_array[i]*state.interestOperatorMultiplierHorizontalInvariance) < state.sadThreshold) {
        if (right_val_array[i] >= sobelLimit && NUMERIC_CONST*state.horizontalInvarianceMultiplier*(float)sad_array[i]/((float)sobel_array[i]) < state.sadThreshold) {
            return true;
        }
    }
    return false;
    '''


    return False

def  GetSAD(leftImage, rightImage, laplacian_left, laplacian_right, j, i, state):
    sadINT = 0
    return sadINT

def pushBroomDispCalc(img1, img2, intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR):
    ############# CALCULATE Disparity ############################

    undistorted_image_L = cv2.undistort(img1, intrinsic_matrixL, distCoeffL, None)
    undistorted_image_R = cv2.undistort(img2, intrinsic_matrixR, distCoeffR, None)

    # --> calculate disparity images
    #disparity_visual = disp.getDisparity(imgLeft=undistorted_image_L, imgRight=undistorted_image_R, method="BM")


    gray_left = cv2.cvtColor(undistorted_image_L, cv2.cv.CV_BGR2GRAY)
    gray_right = cv2.cvtColor(undistorted_image_R, cv2.cv.CV_BGR2GRAY)
    print gray_left.shape
    cols, rows = gray_left.shape
    #if method == "BM":
    '''
    sbm = cv2.cv.CreateStereoBMState()
    disparity = cv2.cv.CreateMat(c, r, cv2.cv.CV_32F)
    sbm.SADWindowSize = 9
    sbm.preFilterType = 1
    sbm.preFilterSize = 5
    sbm.preFilterCap = 61
    sbm.minDisparity = -39
    sbm.numberOfDisparities = 112
    sbm.textureThreshold = 507
    sbm.uniquenessRatio = 0
    sbm.speckleRange = 8
    sbm.speckleWindowSize = 0
    '''

    sbm = cv2.cv.CreateStereoBMState()
    disparity = cv2.cv.CreateMat(cols, rows, cv2.cv.CV_32F)
    sbm.SADWindowSize = 7 #9
    sbm.preFilterType = 1 #1
    sbm.preFilterSize = 5 #5
    sbm.preFilterCap = 21 #61
    sbm.minDisparity = -251 #-39
    sbm.numberOfDisparities =  16*15 #112 # higher the number the less disparities it will find
    sbm.textureThreshold = 600 #507
    sbm.uniquenessRatio = 2 #0
    sbm.speckleRange = 100 #8
    sbm.speckleWindowSize = 10  #0 # decides how many pixels must be close to each other for the algorithm to keep them


    #gray_left = cv2.cv.fromarray(gray_left)
    #gray_right = cv2.cv.fromarray(gray_right)

    cv2.cv.FindStereoCorrespondenceBM(gray_left, gray_right, disparity, sbm)
    disparity_visual = cv2.cv.CreateMat(cols, rows, cv2.cv.CV_8U)

    cv2.cv.Normalize(disparity, disparity_visual, 0, 255, cv2.cv.CV_MINMAX)
    disparity_visual = np.array(disparity_visual)

    return disparity_visual

def pushBroomAlgo(img1,img2,disparity_visual):
    ########### push broom algorithm #####################

    # // make sure that the inputs are of the right type
    img1 = img1.astype(np.float32) # CV_8UC1
    img2 = img2.astype(np.float32)

    cols, rows = img1.shape[:2][::-1]


    # 1 calculate disparity with block matching
    #disparity_visual = disparity_visual.astype(np.float32) # CV_8UC1

    # 2  score blocks based on their abundance of edges

    # 3 create edge map for Left_image and Right_image using the laplacian

    laplacian_leftIMG = cv2.Laplacian(src=img1, ddepth=cv2.CV_64F, ksize=3)
    laplacian_rightIMG = cv2.Laplacian(src=img2, ddepth=cv2.CV_64F, ksize=3)

    # 4 reject any blocks below a threshold for edge-abundance

    '''
    cv::vector<Point3f> pointVector3dArray[NUM_THREADS+1];
    cv::vector<Point3i> pointVector2dArray[NUM_THREADS+1];
    cv::vector<uchar> pointColorsArray[NUM_THREADS+1];

    //cout << "[main] firing worker threads..." << endl;

    if (state.lastValidPixelRow > 0) {

        // crop image to be only include valid pixels
        rows = state.lastValidPixelRow;
    }
    '''


    # 5 score the remaining blocks based on SAD match divided by the
    # summation of edge-values in the pixel block:

    '''
    int numPoints = 0;
    // compute the required size of our return vector
    // this prevents multiple memory allocations
    for (int i=0;i<NUM_THREADS;i++)
    {
        numPoints += pointVector3dArray[i].size();
    }
    pointVector3d->reserve(numPoints);
    pointColors->reserve(numPoints);

    '''


    # for each pixel value i
    #Score = disparityValue_SAD/(pixelValue_5x5(laplacian_leftIMG)+ pixelValue_5x5(laplacian_rightIMG) )

    # 6  REJECTION OF BAD DISPARITY (horizontal self-similarity, such as buildings with gridlike windows or an uninterrupted horizon)

    #  additional blockmatching searches in the right image near our candidate Obstacle.


    #  If we find that one block in the left image matches
    # blocks in the right image at different disparities, we conclude
    # that the pixel block exhibits local self-similarity and reject it.


    #laplacian = cv2.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]])


    #########################################################

### Main method
def Main():
    lastobstacleseen = time.time()
    startTimeMessage = time.time()

    pts_que_center = deque(maxlen=15)
    #pts_que_radius = deque(maxlen=15)

    pts_que_path_center = deque(maxlen=15)
    # Tresh value
    pts_que_center_list = [deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)] # list holds 5 elements
    pts_que_radius_list = [deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)]
    yMove_list = [deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)]

    # tuning parameters
    radiusTresh = 40
    folderName_saveImages = "savedImages"
    toktName = "tokt1"
    object_real_world_mm = 1000 # 1000mm = 1 meter to calculate distance to a known object.
    isObsticleInFrontTreshValue = 1.7
    isObstacleInfront_based_on_radius = False # assusme there is no objects in front when we start the program.

    # Load camerea parameters
    # load calibration parameters
    [intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR] = disp.loadCameraParameters()

     # Parameters
    #b:= base offset, (the distance *between* your cameras)
    base_offset = 30.5

    #f:= focal length of camera,
    fx = 2222
    focal_length = (fx*35)/1360   # 1360 is the width of the image, 35 is width of old camera film in mm (10^-3 m)
    #Distance_map = (base_offset*focal_length)/disparity_visual

    # open files to write to
    pathDirFile = open("pathDir.txt", 'w')
    #with open("pathDir.txt", 'w') as f:

    timeTXTfileName = "timeImages_" + toktName + ".txt"
    #with open(timeTXTfileName, 'w') as f:
    timeTXTfile = open(timeTXTfileName, 'w')


    with Vimba() as vimba:
        system = vimba.getSystem()
        system.runFeatureCommand("GeVDiscoveryAllOnce")

        time.sleep(0.2)
        camera_ids = vimba.getCameraIds()
        for cam_id in camera_ids:
            print "Camera found: ", cam_id

    ################ cam 1  info-------------------------------------------------
        c1 = vimba.getCamera(camera_ids[0])
        c1.openCamera()
        try:
            #gigE camera
            print c1.GevSCPSPacketSize
            print c1.StreamBytesPerSecond  # Bandwidth allocation can be controlled by StreamBytesPerSecond, or by register SCPD0.
            c1.StreamBytesPerSecond = 35000000  # 50000000 #100000000   # gigabyte ethernet cable
        except:
            #not a gigE camera
            pass

        #set pixel format
        # colorFormat
        c1.PixelFormat="BGR8Packed"  # OPENCV DEFAULT  #c0.PixelFormat="Mono8" #

        # give the camera a short break
        time.sleep(0.2)

        #c0.ExposureTimeAbs=60000
        c1.ExposureTimeAbs=100000

        frame1 = c1.getFrame()
        frame1.announceFrame()

        c1.startCapture()

        framecount1 = 0
        droppedframes1 = []

    # ################## cam 2 info -------------------------------------------

        c2 = vimba.getCamera(camera_ids[1])
        c2.openCamera()
        try:
            #gigE camera
            print c2.GevSCPSPacketSize
            print c2.StreamBytesPerSecond
            c2.StreamBytesPerSecond =  35000000  # 50000000 #100000000 --> taking the half og MAXIMUM = 124000000, and gives a margin
            #MINIMUM: 1000000
            #MAXIMUM: 124000000

        except:
            #not a gigE camera
            pass


        c2.PixelFormat="BGR8Packed"  # OPENCV DEFAULT
        # give the camera a short break
        time.sleep(0.2)

        c2.ExposureTimeAbs=100000

        frame2 = c2.getFrame()
        frame2.announceFrame()
        c2.startCapture()

        framecount2 = 0
        droppedframes2 = []


        c1.runFeatureCommand("AcquisitionStart")
        c2.runFeatureCommand("AcquisitionStart")

        ############### Cam 1 and 2 aquire images simultanuasly ##################################################
        # Both camereas will aquire images in this while loop

        start_time = time.time()
        # Setting the times for the first while loop
        plyTime= 2
        dispTime = 0.3
        # pairNumber for saving images that has been used for creating disparity
        pairNumber = 0
        directionTime = 15
        isDirectionDecided = False

        while 1:
            # get status of camera 1
            try:
                frame1.queueFrameCapture()
                success1 = True
            except:
                droppedframes1.append(framecount1)
                success1 = False

            # get status of camera 2
            try:
                frame2.queueFrameCapture()
                success2 = True
            except:
                droppedframes1.append(framecount2)
                success2 = False

            # Set the "shutter speed" --> How long the camera takes in light to make the image
            #c0.runFeatureCommand("ExposureMode=Timed")   # ExposureMode = Timed # PieceWiseLinearHDR
            #c0.runFeatureCommand("ExposureAuto = Continuous")

            # Set the gain-- ISO
            #c0.runFeatureCommand("GainAuto")

            frame1.waitFrameCapture(100)  # 1000
            frame_data1 = frame1.getBufferByteData()

            frame2.waitFrameCapture(100)
            frame_data2 = frame2.getBufferByteData()

            ############################### COMPUTER VISION PART #########################################################################
            if success1 and success2:

                img1 = getImg(frame_data1, frame1)
                img2 = getImg(frame_data2, frame2)

                elapsed_time = time.time() - start_time
                if (elapsed_time > dispTime ):
                    # print "creating disparity"
                    disparity_visual = disparityCalc(img1, img2, intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR)

                    # disparity_visual = pushBroomDispCalc(img1, img2, intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR)
                    #pushBroomAlgo_IMG = pushBroomAlgo(img1,img2,disparity_visual)

                    ####################### Calculations  #########################################

                    # Imporve disparity image, by using a scale sin(20) to sin(50) --> becouse the camera is tilted 35 or 45 degrees?
                    # make an array of values from sin(20) to sin(50)

                    #disparity_visual_adjusted = camereaAngleAdjuster(disparity_visual)

                    # Erode to remove noise
                    IMGbw_PrepcalcCentroid = proc.prepareDisparityImage_for_centroid(disparity_visual)

                    # calculate the centers of the small "objects"
                    image_withCentroids, centerCordinates = proc.findCentroids(IMGbw_PrepcalcCentroid)

                    centerCordinates = np.asarray(centerCordinates) # make list centerCordinates into numpy array

                    #disparity_visualBW = cv2.convertScaleAbs(disparity_visual)

                    #imageDraw, pixelSizeOfObject = proc.drawStuff(centerCordinates, disparity_visual.copy())

                    #image_color_with_Draw, pixelSizeOfObject = proc.drawStuff(centerCordinates, img1.copy())
                    pixelSizeOfObject = 50

                    # calculate the average center of this disparity
                    try:
                        objectAVGCenter = proc.getAverageCentroidPosition(centerCordinates)
                    except:
                        pass
                        isObstacleInfront_based_on_radius = False

                    #object_real_world_mm = 500 # 1000mm = 1 meter
                    distance_mm = proc.calcDistanceToKnownObject(object_real_world_mm, pixelSizeOfObject)
                    #draw.drawTextMessage(image_color_with_Draw, str(distance_mm)) # draw the distance to object

                    # calculate an estimate of distance
                    #print "distance_mm"
                    #print distance_mm

                    disparity_visualBW = cv2.convertScaleAbs(disparity_visual)
                    ####### make image that buffers "old" centerpoints, and calculate center of the biggest centroid -- hopefully that is the biggest object
                    imgStaaker, center, pts_que_center_List, isObstacleInfront_based_on_radius = \
                    findBiggestObject(disparity_visualBW.copy(), pts_que_center, radiusTresh=radiusTresh, isObstacleInfront_based_on_radius=isObstacleInfront_based_on_radius)


                    # draw the lagging of the objects center
                    #image_color_with_Draw = drawTrackedPoints(img1.copy(), pts_que_center_List)

                    #image_color_with_Draw = drawTrackedPoints(image_color_with_Draw.copy(), pts_que_center_List)
                    image_color_with_Draw = drawTrackedPoints(img1.copy(), pts_que_center_List)

                    #cv2.imshow("image after finding minimum bounding rectangle of object", imageDraw )

                    #draw the new center in white
                    centerCircle_Color = (255, 255, 255)
                    try:
                        cv2.circle(disparity_visualBW, objectAVGCenter, 10, centerCircle_Color)
                    except:
                        pass

                    #######################
                    dispTime = (time.time() - start_time) + 0.0035

                    #cv2.imshow("disparity_visual_adjusted", disparity_visual_adjusted)

                    # apply mask so that disparityDisctance() don`t divide by zero
                    #disparity_visual = disparity_visual  #.astype(np.float32) / 16.0

                    #min_disp = 1  # 16
                    #num_disp = 112-min_disp
                    #disparity_visual=(disparity_visual-min_disp)/num_disp
                    #disparity_visual_adjusted_fixed = (disparity_visual_adjusted -min_disp)/num_disp

                    # calculate the Depth_map
                    '''
                    try:
                        Depth_map = disparityDisctance(disparity_visual, focal_length, base_offset)
                    except:
                        pass
                    '''

                    #Depth_map_adjusted = disparityDisctance(disparity_visual_adjusted, focal_length, base_offset)

                    ################################# UDP MESSAGE #########################################
                    # Compare the 3 parts, (Left, Center, Right) with each other to find in what area the object is.
                    #returnValue = compare3windows(depthMap, somthing )

                    #Send position of dangerous objects. To avoid theese positions.
                    # this is for the average position of alle the picels the disparity captures.
                    try:
                        Xpos = proc.findXposMessage(objectAVGCenter)
                        Ypos = proc.findYposMessage(objectAVGCenter)
                        print "Xpos"
                        print Xpos
                    except:
                        isObstacleInfront_based_on_radius = False


                    #XposMessage = directionMessage + ' Xpos :'+ str(Xpos) +' Ypos :' + str(Ypos)
                    #############
                    centerPosMessage = 'Xpos : '+ str(Xpos) +'  Ypos : ' + str(Ypos)


                    directionMessage = "status : "
                    #####
                    '''
                    #if isObsticleInFront(disparity_visual, isObsticleInFrontTreshValue): # if the treshold says there is somthing infront then change directions
                        #directionMessage = obstacleAvoidanceDirection(disparity_visual)
                    if isObstacleInfront_based_on_radius:
                        #directionMessage = "CALC"
                        directionMessage = directionMessage + str(1) + " "

                    else:  # if nothing is in front of camera, do not interupt the path
                        #directionMessage = directionMessage + "CONTINUE"
                        directionMessage = directionMessage + str(0) + " "
                    '''

                    if isObstacleInfront_based_on_radius:
                        directionMessage = directionMessage + str(1) + " "
                        lastobstacleseen = time.time()
                    else:
                        if (time.time()-lastobstacleseen<15):
                            directionMessage = directionMessage + str(1) + " "
                        else:
                            directionMessage = directionMessage + str(0) + " "

                    print "directionMessage"
                    print directionMessage

                    Message = directionMessage + centerPosMessage # only send message if there is a obstacle in front off the viechle

                    ##############################
                    print "#################### Message ###########"
                    print Message
                    sendUDPmessage(Message)

                    #if isObstacleInfront_based_on_radius:
                        #sendUDPmessage(Message)
                    #    sendUDPmessage(MasterMessage)
                    #########
                    try:
                        XposCenterBiggestObject = proc.findXposMessage(center)
                        YposCenterBiggestObject = proc.findYposMessage(center)
                        print "XposCenterBiggestObject"
                        print XposCenterBiggestObject
                    except:
                        isObstacleInfront_based_on_radius = False
                    #XposCenterBiggestObjectMessage = 'XposCenterBiggestObject :'+ str(XposCenterBiggestObject) +'   YposCenterBiggestObject :' + str(YposCenterBiggestObject)

                    #centerPosMessage = 'Xpos : '+ str(XposCenterBiggestObject) +'  Ypos : ' + str(YposCenterBiggestObject)
                    #Message = directionMessage + centerPosMessage
                    #sendUDPmessage(Message)
                    ####

                    # Distances
                    print "distance_mm"
                    print distance_mm

                    if int(Xpos)>0:
                        print "turn right"
                        #direction_string = "turn RIGHT"
                        direction_string = "-->"
                        Xpath = 1100
                        print Xpath

                    if int(Xpos)<0:
                        print "turn left"
                        #direction_string = "turn LEFT"
                        direction_string = "<--"
                        Xpath = 100
                        print Xpath

                    if not isObstacleInfront_based_on_radius:
                        direction_string = "---"

                    draw.drawTextMessage(image_color_with_Draw, direction_string) # shows an arrow if we shoud go left or right

                    CORD = (Xpath,Ypos)
                    disparity_visualBW = proc.drawPath(Xpath,Ypos, disparity_visualBW)

                    #pts_que_center_List
                    pts_que_path_center_List = trackPathPos(pts_que_path_center, Xpath, Ypos)

                    # draw the lagging of the path center
                    #image_color_with_Draw = drawTrackedPoints(img1.copy(), pts_que_path_center_List)
                    image_color_with_Draw = drawTrackedPoints(image_color_with_Draw.copy(), pts_que_path_center_List)

                    # save the planed path
                    print  "path direction in pixel values" + str(CORD)
                    path_string = "path direction in pixel values" + str(CORD) + "\n"
                    print "saving pointCloud"

                    pathDirFile.write(path_string)

                    ############## save the images that has been used to create disparity######################
                    pairNumber = pairNumber + 1
                    imgNameString_L = folderName_saveImages + "/" + toktName + "_L_" + str(pairNumber) + ".jpg"
                    imgNameString_R = folderName_saveImages + "/" + toktName + "_R_" + str(pairNumber) + ".jpg"

                    imgNameString_DISTANCE = folderName_saveImages + "/" + toktName + "_Depth_map_" + str(pairNumber) + ".jpg"

                    imgNameString_DISPARITY = folderName_saveImages + "/" + toktName + "_Disp_map_" + str(pairNumber) + ".jpg"

                    # writing the images to diske
                    cv2.imwrite(imgNameString_L, img1)
                    cv2.imwrite(imgNameString_R, img2)
                    #cv2.imwrite(imgNameString_DISTANCE, Depth_map)
                    cv2.imwrite(imgNameString_DISPARITY, disparity_visual)

                    # write the time theese images have been taken to a file
                    dateTime_string = unicode(datetime.datetime.now())
                    path_string = str(pairNumber) + " , " + str(dateTime_string)
                    print "saving timeImages.txt"
                    # timeImages tokt name must be added
                    timeTXTfile.write(path_string + '\n')

                    ############### DISPLAY IMAGES HERE TO the USER ############################

                    #if you want to se left and right image
                    #dispalyToUser(img1,img2)

                    # if you want to see disparity image

                    # if you want to see adjusted disparity image

                    # if you want to see the drawing on top of objects
                    #cv2.imshow("image after finding minimum bounding rectangle of object", imageDraw )

                    # if you want to view the center of object
                    cv2.imshow("disparity_visualBW disparity_visualBW", disparity_visualBW)

                    #print "drawing over color image"
                    cv2.imshow("color image with drawings", image_color_with_Draw)


                    # if display disparity_visual_adjusted
                    #cv2.imshow("disparity_visual_adjusted", disparity_visual_adjusted)

                    # if display depth map
                    #cv2.imshow("Depth_map", Depth_map)


                    #cv2.imshow("image imgGrowing", imgGrowing )
                    #cv2.imshow('center object', disparity_visual)


                    #######################################################################################
                    '''
                    if (elapsed_time > plyTime):
                        print "creating point cloud"
                        point_cloud(disparity_visual, img1, focal_length)
                        # extend time
                        plyTime = (time.time() - start_time) + 4 #  elapsed_time = 2 seconds

                    # TODO save pointclouds at an acaptable timefrequency
                    elapsed_time = time.time() - start_time
                    '''

                ##################### END PROGRAM CODE ############################################

            framecount1+=1
            print "Frames displayed: %i"%framecount1
            k = cv2.waitKey(1)
            if k == 0x1b:
                cv2.destroyAllWindows()
                print "Frames displayed: %i"%framecount1
                print "Frames dropped: %s"%droppedframes1
                break

            framecount2+=1
            k = cv2.waitKey(1)
            if k == 0x1b:
                cv2.destroyAllWindows()
                print "Frames displayed: %i"%framecount2
                print "Frames dropped: %s"%droppedframes2
                break

        # close the .txt files that had been written to
        pathDirFile.close()
        timeTXTfile.close()

        ############ CLOSE THE CAMEREAS       ############

        c1.runFeatureCommand("AcquisitionStop")
        c2.runFeatureCommand("AcquisitionStop")


        c1.endCapture()
        c1.revokeAllFrames()
        c1.closeCamera()

        c2.endCapture()
        c2.revokeAllFrames()
        c2.closeCamera()

if __name__ == '__main__':
    # cProfile makes it possible for us to analyse the time each function uses
    cProfile.run('Main()')

    # values to change in the program


    # Main()
