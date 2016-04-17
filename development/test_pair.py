
from pymba import *
import numpy as np
import cv2
import time
import cProfile
import math

import socket # to send UDP message to labview pc
import stereo
import colorMethods as color

import disparityMapCalc as disp
import processesBWdisparityIMG as find



def _displayDepth(name, mat):
    s = v = (np.ones(mat.shape) * 255).astype(np.uint8)
    h = ((mat - np.nanmin(mat)) / (np.nanmax(mat) - np.nanmin(mat)) * 255).astype(np.uint8)
    cv2.imshow(name, cv2.cvtColor(cv2.merge([h, s, v]), cv2.cv.CV_HSV2BGR))


def _nothing(_):
    pass

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

def disparityCalc(img1, img2, intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR):
    ############# CALCULATE Disparity ############################

    # TODO: Coloradjust images
    #img1 = color.claheAdjustImages(img1)
    #img2 = color.claheAdjustImages(img2)

    #img1 = color.blocks_clahe(img1)
    #img2 = color.blocks_clahe(img2)

    #--->Undistort images
    # print('Step 3:  and rectify the original images')

    # First undistort images taken with left camera #TODO: Be sure of wich camerea is left and right!!
    #print('Undistort the left images')
    undistorted_image_L = disp.UndistortImage(img1, intrinsic_matrixL, distCoeffL)

    # Secondly undistort images taken with right camera
    #print('Undistort the right images')
    undistorted_image_R = disp.UndistortImage(img2, intrinsic_matrixR, distCoeffR)

    # --> calculate disparity images
    disparity_visual = disp.getDisparity(imgLeft=undistorted_image_L, imgRight=undistorted_image_R, method="BM")  #method="BM")

    return disparity_visual

def disparityDisctance(disparity_visual, focal_length, base_offset):
    #D:= Distance of point in real world,
    #b:= base offset, (the distance *between* your cameras)
    #f:= focal length of camera,
    #d:= disparity:

    #D = b*f/d
    Depth_map = (base_offset*focal_length)/disparity_visual
    return Depth_map

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
    # TODO: test if this gives correct results
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

'''

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


def isObsticleInFront(img):
    meanValue = meanPixelSUM(img)
    print "meanValue for disparity image"
    print meanValue
    if meanValue < 1.7:
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



########
# MAIN #
def main():

    # set the images you want to test here
    imgLeft = r"testImages\obstacle1\LeftCameraRun4_211.png"
    imgRight = r"testImages\obstacle1\RightCameraRun4_211.png"

    frame_left = cv2.imread(imgLeft)
    frame_right = cv2.imread(imgRight)

    cv2.imshow("frame_left", frame_left)
    cv2.imshow("frame_right", frame_right)
    cv2.waitKey(0)

    ########################################################################

    # set variables2
    start_time = time.time()
    # Setting the times for the first while loop
    plyTime= 2
    dispTime = 0.3
    # pairNumber for saving images that has been used for creating disparity
    pairNumber = 0
    folderName_saveImages = "savedImages"
    toktName = "tokt1"

    # Load camerea parameters
    # load calibration parameters
    [intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR] = disp.loadCameraParameters()

    #b:= base offset, (the distance *between* your cameras)
    base_offset = 30.5

    #f:= focal length of camera,
    fx = 2222
    focal_length = (fx*35)/1360
    #Distance_map = (base_offset*focal_length)/disparity_visual

    # start program
    img1 = frame_left
    img2 = frame_right

        # Calculatem edge detection
    # TODO: create canny edge -- decider, if there is to "little" edges in the image, then dont process the image furter...

    #edge1 = cv2.Canny(img1,25,75)
    #edge2 = cv2.Canny(img2,25,75)

    #cv2.imshow("edge1", edge1)
    #cv2.imshow("edge2", edge2)
    #cv2.waitKey(0)


    elapsed_time = time.time() - start_time
    #if (elapsed_time > dispTime ):
    print "creating disparity"
    disparity_visual = disparityCalc(img1, img2, intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR)
    # display disparity map
    print "displaying disparity"
    #cv2.imshow("disparity_visual", disparity_visual)
    #cv2.waitKey(0)
    #cv2.imshow("disparityMovie", disparity_visual)
    dispTime = (time.time() - start_time) + 0.0035

    # Imporve disparity image, by using a scale sin(20) to sin(50) --> becouse the camera is tilted 35 or 45 degrees?
    # make an array of values from sin(20) to sin(50)
    disparity_visual_adjusted = camereaAngleAdjuster(disparity_visual)
    cv2.imshow("disparity_visual_adjusted", disparity_visual_adjusted)
    #cv2.waitKey(0)


    # apply mask so that disparityDisctance() don`t divide by zero
    #disparity_visual = disparity_visual  #.astype(np.float32) / 16.0
    #TODO: doobble check this--> i think i need it to not divide by zero
    min_disp = 1  # 16
    num_disp = 112-min_disp
    #disparity_visual=(disparity_visual-min_disp)/num_disp
    disparity_visual_adjusted = (disparity_visual_adjusted -min_disp)/num_disp


    # FIND the object in the disparity image #############################################
    # disparity_visual_adjusted_Centroid, ctr = findCentroifOfObject(disparity_visual_adjusted)
    #print "ctr"
    #print ctr
    #cv2.imshow("disparity_visual_adjusted_Centroid", disparity_visual_adjusted_Centroid)
    #cv2.waitKey(0)

    disparity_visual_adjusted_Centroid2, centerCordinates = find.findCentroids( disparity_visual)#disparity_visual)
    cv2.imshow('disparity_visual_adjusted_Centroid2', disparity_visual_adjusted_Centroid2)
    0xFF & cv2.waitKey()

    objectCenter = find.getAverageCentroidPosition(centerCordinates)

    #Draw the new center in white and display image
    centerCircle_Color = (255, 255, 255)
    cv2.circle(disparity_visual_adjusted_Centroid2, objectCenter, 10, centerCircle_Color)
    cv2.imshow('center object', disparity_visual_adjusted_Centroid2)
    cv2.waitKey(0)

    # USE objectCenter and disparity image to calculate distance to this point.

    ########## NEED poincloud inforation to know distance to centroid

    points3D = reconstructScene(disparityMap, stereoParams);

    # Convert to meters and create a pointCloud object
    #points3D = points3D ./ 1000;


    np.ravel_multi_index()
    # Find the 3-D world coordinates of the centroids.
    #centroidsIdx = np.ravel_multi_index(np.shape(disparity_visual), centroids(:, 2), centroids(:, 1));
    centroidsIdx = np.ravel_multi_index(np.shape(disparity_visual), centerCordinates[:, 2], centerCordinates[:, 1])
    X = points3D(:, :, 1);
    Y = points3D(:, :, 2);
    Z = points3D(:, :, 3);
    #centroids3D = [X(centroidsIdx)'; Y(centroidsIdx)'; Z(centroidsIdx)'];
    centroids3D = [X(centroidsIdx)T Y(centroidsIdx)T Z(centroidsIdx)T]




    # calculate the Depth_map
    Depth_map = disparityDisctance(disparity_visual_adjusted, focal_length, base_offset )

    print "displaying Depth_map"
    cv2.imshow("Depth_map", Depth_map)
    cv2.waitKey(0)

    '''
    print "creating point cloud"
    ply_string = stereo.point_cloud(disparity_visual, img1, focal_length)
    print "saving pointCloud"
    with open("set.ply", 'w') as f:
        f.write(ply_string)
    '''

    # todo: filter out everything further away than 2 meter to test

    # Save the images that has been used to create disparity
    pairNumber = pairNumber + 1
    imgNameString_L = folderName_saveImages + "/" + toktName + "_L_" + str(pairNumber) + ".jpg"
    imgNameString_R = folderName_saveImages + "/" + toktName + "_R_" + str(pairNumber) + ".jpg"

    imgNameString_DISTANCE = folderName_saveImages + "/" + toktName + "_Depth_map_" + str(pairNumber) + ".jpg"

    # writing the images to disk
    cv2.imwrite(imgNameString_L, img1)
    cv2.imwrite(imgNameString_R, img2)
    cv2.imwrite(imgNameString_DISTANCE, disparity_visual)#Depth_map)


    # Compare the 3 parts, (Left, Center, Right) with each other to find in what area the object is.
    #returnValue = compare3windows(depthMap, somthing )
    # Image ROI
    if isObsticleInFront(disparity_visual):
        directionMessage = obstacleAvoidanceDirection(disparity_visual)
    else:
        directionMessage = "CONTINUE"


    print "directionMessage"
    print directionMessage

    #if (elapsed_time > plyTime):
    #    print "creating point cloud"
    #    ply_string = stereo.point_cloud(disparity_visual, img1 ,focal_length)
    #    print "saving pointCloud"
    #    with open("set.ply", 'w') as f:
    #        f.write(ply_string)

    #    plyTime = (time.time() - start_time) + 4 #  elapsed_time = 2 seconds

    # TODO: Send UDP message to pc running labview program
    #stringMessage = "hei hei du er kul"
    #sendUDPmessage(stringMessage)

    ###########################################################################

    '''
    [intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR] = loadCameraParameters()

    # Create dst, map_x and map_y with the same size as src:
    #dst.create( src.size(), src.type() );
    map_x_left = np.zeros( frame_left.shape[:2],np.float32 )
    map_y_left = np.zeros( frame_left.shape[:2],np.float32  )

    map_x_right = np.zeros( frame_right.shape[:2],np.float32 )
    map_y_right = np.zeros( frame_right.shape[:2],np.float32 )

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


    frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    #frame_left = cv2.remap(frame_left,  map_x_left,  map_y_left, cv2.INTER_LINEAR)
    #frame_right = cv2.remap(frame_right, map_x_right, map_y_right, cv2.INTER_LINEAR)
    #cv2.imshow('left', frame_left)
    #cv2.imshow('right', frame_right)
    disparity = stereo.compute(frame_left,
                               frame_right).astype(np.float32) / 16
    disparity = np.uint8(disparity)
    #disparity = np.float32(disparity)
    #_displayDepth('tuner', disparity)
    cv2.imshow('tuner', disparity)
    cv2.imshow('left', frame_left)
    cv2.imshow('right', frame_right)
    cv2.waitKey(0)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # Update based on GUI values
    tuner_minDisparity = cv2.getTrackbarPos('minDisparity', 'tuner')
    tuner_numDisparities = max((cv2.getTrackbarPos('numDisparities', 'tuner') / 16) * 16, 16)
    tuner_SADWindowSize = cv2.getTrackbarPos('SADWindowSize', 'tuner')
    tuner_P1 = cv2.getTrackbarPos('P1', 'tuner')
    tuner_P2 = cv2.getTrackbarPos('P2', 'tuner')

    stereo = cv2.StereoSGBM(tuner_minDisparity, tuner_numDisparities, tuner_SADWindowSize,
                            tuner_P1, tuner_P2, tuner_disp12MaxDiff)

    # Get the next frame before attempting to run this loop again
    ret_left, frame_left = left_video.read()
    ret_right, frame_right = right_video.read()

    # Destroy all windows
    cv2.destroyAllWindows()
    '''

if __name__ == '__main__':
    main()