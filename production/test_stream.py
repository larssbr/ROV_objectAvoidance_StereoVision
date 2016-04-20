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
import datetime # to print the time to the textfile
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
    UDP_IP = "127.0.0.1"
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

    cxR = 637.64260
    cyR = -33.60849
    cxL = 681.42537
    cyL = -22.08306

    Cx = (cxR + cxL)/2
    Cy = abs(cyR + cyL)/2

    Tx = 30.5  #Tx is the distance between the two camera lens focal centers
    a = 1/Tx   #, where Tx is the distance between the two camera lens focal centers
    # b = (cx -cx)/tx I think this compensates for misalighmen

    b = (cxR-cxL)/Tx
    b = 0


    Q = np.float32([[1, 0, 0, -Cx],
                    [0, 1, 0,  -Cy],
                    [0, 0, 0, focal_length],
                    [0, 0, a, b]])

    #Using the recomendations
    focal_length = 0.8 * w
    Q = np.float32([[1, 0, 0, -Cx],
                [0, -1, 0,  Cy],
                [0, 0, 0, -focal_length],
                [0, 0, 1, 0]])

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

def findBiggestObject(img, pts_que_center, pts_que_radius, radiusTresh=40):
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
        M = cv2.moments(c)
        biggestObjectCenter = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > radiusTresh: # works as a treshold
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(img, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(img, biggestObjectCenter, 5, (0, 0, 255), -1)

    # update the points queue
    pts_que_center.appendleft(biggestObjectCenter)
    pts_que_radius.appendleft(radius)
    pts_que_center_List = list(pts_que_center)
    pts_que_radius_List = list(pts_que_radius)


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

    return img, biggestObjectCenter, pts_que_center_List, pts_que_radius_List

def _displayDepth(name, mat):
    s = v = (np.ones(mat.shape) * 255).astype(np.uint8)
    h = ((mat - np.nanmin(mat)) / (np.nanmax(mat) - np.nanmin(mat)) * 255).astype(np.uint8)
    cv2.imshow(name, cv2.cvtColor(cv2.merge([h, s, v]), cv2.cv.CV_HSV2BGR))

def loadCameraParameters():
    # left
    fxL = 2222.72426
    fyL = 2190.48031

    k1L =  0.27724
    k2L = 0.28163
    k3L = -0.06867
    k4L = 0.00358
    k5L = 0.00000

    cxL = 681.42537
    cyL = -22.08306

    skewL=0

    # todo: find the p values
    p1L=0
    p2L=0
    p3L=0
    p4L=0

    # right
    fxR = 2226.10095
    fyR = 2195.17250


    k1R =  0.29407
    k2R = 0.29892
    k3R = 0-0.08315
    k4R = -0.01218
    k5R = 0.00000

    cxR = 637.64260
    cyR = -33.60849


    skewR=0


    # todo: find the p values
    p1R=0
    p2R=0
    p3R=0
    p4R=0

    # x0 and y0 is zero
    x0=0
    y0=0

    intrinsic_matrixL = np.matrix([[fxL, skewL , x0], [0, fyL, y0], [0, 0, 1]])
    intrinsic_matrixR = np.matrix([[fxR, skewR , x0], [0, fyR, y0], [0, 0, 1]])

    distCoeffL = np.matrix([k1L, k2L, p1L, p2L, k3L])
    distCoeffR = np.matrix([k1R, k2R, p1R, p2R, k3R])
    return intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR

############# post process disparity image ##################
'''
def initDataCost(imgLeft, imgLeft, mrf):
    # Cache the data cost results so we don't have to recompute it every time

    # Force greyscale
    # Make imgLeft and right grayscale here

    assert(left.channels() == 1);

    mrf.width = left.cols;
    mrf.height = left.rows;

    int total = mrf.width*mrf.height;

    mrf.grid.resize(total);

    # Initialise all messages to zero
    for(int i=0; i < total; i++):
        for(int j=0; j < 5; j++):
            for(int k=0; k < LABELS; k++):
                mrf.grid[i].msg[j][k] = 0;


    #Add a border around the image
    int border = LABELS;

    for(int y=border; y < mrf.height-border; y++):
        for(int x=border; x < mrf.width-border; x++):
            for(int i=0; i < LABELS; i++):
                mrf.grid[y*left.cols+x].msg[DATA][i] = DataCostStereo(left, right, x, y, i)


def improveDisparity(imgLeft, imgRight,img):
    # innspiration from http://nghiaho.com/?page_id=1366

    #  We expect pixels near each other to have similar disparity, unless there is a genuine boundary.
    width, height = img.shape[:2][::-1]
    totalPX = width*height

    # parameters, specific to dataset
    BP_ITERATIONS = 40
    LABELS = 16
    LAMBDA = 20
    SMOOTHNESS_TRUNC = 2

    initDataCost(imgLeft, imgLeft, mrf)

################# end post process disparity image  ########################################

def stereoMatcherTuner():
     # NOT TESTED YET>>>>>>>>>
        ################# TUNDER FOR disparity ###############################

        # StereoSGBM values
    tuner_minDisparity = 10
    tuner_numDisparities = 128
    tuner_SADWindowSize = 9
    tuner_P1 = 8 * 3 * 9 * 9
    tuner_P2 = 32 * 3 * 9 * 9
    tuner_disp12MaxDiff = -1

        # Block matcher
    stereo = cv2.StereoSGBM(tuner_minDisparity, tuner_numDisparities, tuner_SADWindowSize,
                            tuner_P1, tuner_P2, tuner_disp12MaxDiff)




    # Tuner GUI
    cv2.namedWindow('tuner')
    cv2.createTrackbar('minDisparity', 'tuner', tuner_minDisparity, 100, _nothing)
    cv2.createTrackbar('numDisparities', 'tuner', tuner_numDisparities, 2048, _nothing)
    cv2.createTrackbar('SADWindowSize', 'tuner', tuner_SADWindowSize, 19, _nothing)
    cv2.createTrackbar('P1', 'tuner', tuner_P1, 5000, _nothing)
    cv2.createTrackbar('P2', 'tuner', tuner_P2, 100000, _nothing)


    # Update StereoSGBM values based on GUI values
    tuner_minDisparity = cv2.getTrackbarPos('minDisparity', 'tuner')
    tuner_numDisparities = max((cv2.getTrackbarPos('numDisparities', 'tuner') / 16) * 16, 16)
    tuner_SADWindowSize = cv2.getTrackbarPos('SADWindowSize', 'tuner')
    tuner_P1 = cv2.getTrackbarPos('P1', 'tuner')
    tuner_P2 = cv2.getTrackbarPos('P2', 'tuner')

    stereo = cv2.StereoSGBM(tuner_minDisparity, tuner_numDisparities, tuner_SADWindowSize,
                            tuner_P1, tuner_P2, tuner_disp12MaxDiff)


'''
    #######################################################################

def sendUDPmessage(MESSAGE):

    # addressing information of target
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5005
    #MESSAGE = "Hello, World!"

    print "UDP target IP:", UDP_IP
    print "UDP target port:", UDP_PORT
    print "message:", MESSAGE

    # initialize a socket, think of it as a cable
    # SOCK_DGRAM specifies that this is UDP
    sock = socket.socket(socket.AF_INET, # Internet
                         socket.SOCK_DGRAM) # UDP
     # send the command
    sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))

    # close the socket
    sock.close()

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

#def treshPixelSUM(imgROI):
    #threshold estimation method

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
    if meanValue < isObsticleInFrontTreshValue:
        return True
    else:
        return False

# Fourier Transform
def FourierMethod(img, method):

    rows, cols = img.shape[:2][::-1]
    crow, ccol = rows/2 , cols/2

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    if method == 1:

        fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)


    ########
    if method == 2:
        dft = cv2.dft(np.float32(img), flags=cv2.nonzeroRows    )  #flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Create a mask first, center square is 1, remaining all zeros
        mask = np.zeros((rows,cols,2),np.uint8)
        mask[crow-30:crow+30, ccol-30:ccol+30] = 1

        # apply mask and inverse DFT
        fshift = dft_shift*mask
        #fshift =dft*mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

    #############

    return img_back

def initialize():

    pts_que_center = deque(maxlen=15)
    pts_que_radius = deque(maxlen=15)
    # Tresh value
    pts_que_center_list = [deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)] # list holds 5 elements
    pts_que_radius_list = [deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)]
    yMove_list = [deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)]

    # tuning parameters
    radiusTresh = 40
    folderName_saveImages = "savedImages"
    toktName = "tokt1"
    object_real_world_mm = 500 # 1000mm = 1 meter to calculate distance to a known object.
    isObsticleInFrontTreshValue = 1.7

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

def nothing(x):
    pass

def methodEN(img1, img2):
    start_time = time.time()
    #img1 = getImg(frame_data1, frame1)
    #img2 = getImg(frame_data2, frame2)
    dispTime = 0
    #[intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR] = disp.loadCameraParameters()
    pairNumber =0
    ##############
    pts_que_center = deque(maxlen=15)
    pts_que_radius = deque(maxlen=15)
    # Tresh value
    pts_que_center_list = [deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)] # list holds 5 elements
    pts_que_radius_list = [deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)]
    yMove_list = [deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)]

    # tuning parameters
    ##############################
    trackBarWindowName = 'image'
    cv2.namedWindow(trackBarWindowName) # name of window to TUNE

    # create trackbars for color change
    #cv2.createTrackbar('R',trackBarWindowName,0,255,nothing)
    #cv2.createTrackbar('G',trackBarWindowName,0,255,nothing)
    #cv2.createTrackbar('B',trackBarWindowName,0,255,nothing)
    cv2.createTrackbar('radiusTresh', trackBarWindowName, 0, 100, nothing)

    #radiusTresh = cv2.getTrackbarPos('radiusTresh', trackBarWindowName)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, trackBarWindowName, 0, 1, nothing)

    ##################################

    radiusTresh = 40
    folderName_saveImages = "savedImages"
    toktName = "tokt1"
    object_real_world_mm = 500 # 1000mm = 1 meter to calculate distance to a known object.
    isObsticleInFrontTreshValue = 1.7

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

    ############

    elapsed_time = time.time() - start_time +1
    if (elapsed_time > dispTime ):
        # print "creating disparity"
        disparity_visual = disparityCalc(img1, img2, intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR)

        # disparity_visual = pushBroomDispCalc(img1, img2, intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR)
        #pushBroomAlgo_IMG = pushBroomAlgo(img1,img2,disparity_visual)

        ####################### Calculations  #########################################
        #proc

        # Imporve disparity image, by using a scale sin(20) to sin(50) --> becouse the camera is tilted 35 or 45 degrees?
        # make an array of values from sin(20) to sin(50)
        disparity_visual_adjusted = camereaAngleAdjuster(disparity_visual)

        disparity_visual = disparity_visual.astype(np.float32)

        # Erode to remove noise
        IMGbw_PrepcalcCentroid = proc.prepareDisparityImage_for_centroid(disparity_visual)

        # calculate the centers of the small "objects"
        image_withCentroids, centerCordinates = proc.findCentroids(IMGbw_PrepcalcCentroid)

        centerCordinates = np.asarray(centerCordinates) # make list centerCordinates into numpy array

        #imageDraw, pixelSizeOfObject = proc.drawStuff(centerCordinates, disparity_visual.copy())

        #image_color_with_Draw, pixelSizeOfObject = proc.drawStuff(centerCordinates, img1.copy())
        pixelSizeOfObject = 50

        # calculate the average center of this disparity
        objectAVGCenter = proc.getAverageCentroidPosition(centerCordinates)

        #object_real_world_mm = 500 # 1000mm = 1 meter
        distance_mm = proc.calcDistanceToKnownObject(object_real_world_mm, pixelSizeOfObject)
        #draw.drawTextMessage(image_color_with_Draw, str(distance_mm)) # draw the distance to object
        image_color_with_Draw = img1.copy()
        draw.drawTextMessage(image_color_with_Draw, str(distance_mm))
        # calculate an estimate of distance
        #print "distance_mm"
        #print distance_mm

        print "disparity_visual.dtype"
        print disparity_visual.dtype # float32

        cv2.imshow("disparity_visual", disparity_visual)
        #cv2.waitKey(0)

        #print "drawing over color image"
        cv2.imshow("image_color_with_Draw",image_color_with_Draw)
        #cv2.waitKey(0)


        disparity_visualBW = cv2.convertScaleAbs(disparity_visual)
        #and if you do : print(disparity.dtype) it shows : int16 and  it shows : uint8. so the type of the image has changed.

        print "(res.dtype)"
        print(disparity_visual.dtype)

        #print "disp.dtype"
        #print disp.dtype # float32

        #cv2.imshow("disparity_visual normailized", disparity_visualBW)
        #cv2.waitKey(0)

        # update radiusTresh with tuner
        radiusTresh = cv2.getTrackbarPos('radiusTresh', trackBarWindowName)
        ####### make image that buffers "old" centerpoints, and calculate center of the biggest centroid -- hopefully that is the biggest object
        imgStaaker, center, pts_que_center_List, pts_que_radius_List = findBiggestObject(disparity_visualBW.copy(), pts_que_center, pts_que_radius, radiusTresh=radiusTresh)

        #cv2.imshow("image after finding minimum bounding rectangle of object", imageDraw )

        #draw the new center in white
        centerCircle_Color = (255, 255, 255)
        cv2.circle(imgStaaker, objectAVGCenter, 10, centerCircle_Color)

        #######################
        dispTime = (time.time() - start_time) + 0.0035


        #cv2.imshow("disparity_visual_adjusted", disparity_visual_adjusted)


        # apply mask so that disparityDisctance() don`t divide by zero
        #disparity_visual = disparity_visual  #.astype(np.float32) / 16.0
        #TODO: doobble check this--> i think i need it to not divide by zero
        #min_disp = 1  # 16
        #num_disp = 112-min_disp
        #disparity_visual=(disparity_visual-min_disp)/num_disp
        #disparity_visual_adjusted_fixed = (disparity_visual_adjusted -min_disp)/num_disp

        # calculate the Depth_map
        try:
            Depth_map = disparityDisctance(disparity_visual, focal_length, base_offset)
        except:
            pass

        #Depth_map_adjusted = disparityDisctance(disparity_visual_adjusted, focal_length, base_offset)

        ################################# UDP #########################################
        # Compare the 3 parts, (Left, Center, Right) with each other to find in what area the object is.
        #returnValue = compare3windows(depthMap, somthing )
        # Image ROI
        ####
        directionMessage = "status : "
        #####
        if isObsticleInFront(disparity_visual, isObsticleInFrontTreshValue): # if the treshold says there is somthing infront then change directions
            #directionMessage = obstacleAvoidanceDirection(disparity_visual)

            #directionMessage = "CALC"
            directionMessage = directionMessage + str(0) + " "
        else:  # if nothing is in front of camera, do not interupt the path
            #directionMessage = directionMessage + "CONTINUE"
            directionMessage = directionMessage + str(1) + " "

        print "directionMessage"
        print directionMessage


        #Send position of dangerous objects. To avoid theese positions.
        # this is for the average position of alle the picels the disparity captures.
        Xpos = proc.findXposMessage(objectAVGCenter)
        Ypos = proc.findYposMessage(objectAVGCenter)
        print "Xpos"
        print Xpos
        #XposMessage = directionMessage + ' Xpos :'+ str(Xpos) +' Ypos :' + str(Ypos)
        #############
        centerPosMessage = 'Xpos : '+ str(Xpos) +'  Ypos : ' + str(Ypos)
        Message = directionMessage + centerPosMessage
        sendUDPmessage(Message)
        #########

        XposCenterBiggestObject = proc.findXposMessage(center)
        YposCenterBiggestObject = proc.findYposMessage(center)
        print "XposCenterBiggestObject"
        print XposCenterBiggestObject
        #XposCenterBiggestObjectMessage = 'XposCenterBiggestObject :'+ str(XposCenterBiggestObject) +'   YposCenterBiggestObject :' + str(YposCenterBiggestObject)
        ######## TODO: test this method under water
        #centerPosMessage = 'Xpos : '+ str(XposCenterBiggestObject) +'  Ypos : ' + str(YposCenterBiggestObject)
        #Message = directionMessage + centerPosMessage
        #sendUDPmessage(Message)
        ####

        # Distances
        print "distance_mm"
        print distance_mm


        if Xpos>0:
            print "turn right"
            Xpath = 1100
            print Xpath
        else:
            print "turn left"
            Xpath = 100
            print Xpath

        CORD = (Xpath,Ypos)
        imgStaaker = proc.drawPath(Xpath,Ypos, imgStaaker)

        ############## save the images that has been used to create disparity######################
        pairNumber = pairNumber + 1
        imgNameString_L = folderName_saveImages + "/" + toktName + "_L_" + str(pairNumber) + ".jpg"
        imgNameString_R = folderName_saveImages + "/" + toktName + "_R_" + str(pairNumber) + ".jpg"

        imgNameString_DISTANCE = folderName_saveImages + "/" + toktName + "_Depth_map_" + str(pairNumber) + ".jpg"

        imgNameString_DISPARITY = folderName_saveImages + "/" + toktName + "_Disp_map_" + str(pairNumber) + ".jpg"

        # writing the images to diske
        cv2.imwrite(imgNameString_L, img1)
        cv2.imwrite(imgNameString_R, img2)
        cv2.imwrite(imgNameString_DISTANCE, Depth_map)
        cv2.imwrite(imgNameString_DISPARITY, disparity_visual)


        # write the time theese images have been taken to a file
        dateTime_string = unicode(datetime.datetime.now())
        path_string = str(pairNumber) + " , " + str(dateTime_string)
        print "saving timeImages.txt"
        # timeImages tokt name must be added
        timeTXTfileName = "timeImages_" + toktName + ".txt"

        with open(timeTXTfileName, 'w') as f:
            f.write(path_string + '\n')

        ############### DISPLAY IMAGES HERE TO the USER ############################

        #if you want to se left and right image
        #dispalyToUser(img1,img2)

        # if you want to see disparity image


        # if you want to see adjusted disparity image

        # if you want to see the drawing on top of objects
        #cv2.imshow("image after finding minimum bounding rectangle of object", imageDraw )


        CORD = (Xpath,Ypos)
        print  "path direction in pixel values" + str(CORD)
        path_string = "path direction in pixel values" + str(CORD)
        print "saving pathDir.txt"
        with open("pathDir.txt", 'w') as f:
            f.write(path_string + '\n')

        imgStaaker = proc.drawPath(Xpath, Ypos, imgStaaker)

        # if you want to view the center of object
        cv2.imshow(trackBarWindowName, imgStaaker)
        cv2.waitKey(0)

        # if display disparity_visual_adjusted
        #cv2.imshow("disparity_visual_adjusted", disparity_visual_adjusted)

        # if display depth map
        #cv2.imshow("Depth_map", Depth_map)


        #cv2.imshow("image imgGrowing", imgGrowing )
        #cv2.imshow('center object', disparity_visual)


        #######################################################################################

        #print "creating point cloud"
        #point_cloud(disparity_visual, img1, focal_length)
        #print "saving pointCloud"

        #with open("set.ply", 'w') as f:
        #   f.write(ply_string)


        time.sleep(4)

        '''
        if (elapsed_time > plyTime):
            print "creating point cloud"
            point_cloud(disparity_visual, img1, focal_length)
            # extend time
            plyTime = (time.time() - start_time) + 4 #  elapsed_time = 2 seconds

        # TODO save pointclouds at an acaptable timefrequency
        elapsed_time = time.time() - start_time
        '''

        #cv2.imshow("centroid_img", centroid_img)

        ##################### END PROGRAM CODE ############################################


########
# MAIN #
def main():

    # set the images you want to test here
    imgLeft = r"testImages\obstacle1\LeftCameraRun4_211.png"
    imgRight = r"testImages\obstacle1\RightCameraRun4_211.png"

    #imgLeft = r"savedImages\tokt1_L_786.jpg"
    #imgRight = r"savedImages\tokt1_R_786.jpg"

    frame_left = cv2.imread(imgLeft)
    frame_right = cv2.imread(imgRight)

    #cv2.imshow("frame_left", frame_left)
    #cv2.imshow("frame_right", frame_right)
    #cv2.waitKey(0)

    ########################################################################
    initialize()

    methodEN(frame_left, frame_right)


if __name__ == '__main__':
    main()