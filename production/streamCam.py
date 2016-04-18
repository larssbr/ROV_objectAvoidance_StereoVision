
"""
Lars Brusletto
This code works. On 4/09/2016
It captures a stream from 2 stereo camereas

Then it creates a disparity map

TODO

Click a button and create a .ply file to visualize depth
or for each 2 seconds

 ---> Calculate distance -

"""

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


#import stereo as stereo

def printTime():
    time.time()

def disparityCalc(img1, img2, intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR):
    ############# CALCULATE Disparity ############################

    # TODO: Coloradjust images

    #--->Undistort images
    # print('Step 3:  and rectify the original images')

    # First undistort images taken with left camera
    #TODO: Be sure of wich camerea is left and right!!





    #print('Undistort the left images')
    undistorted_image_L = disp.UndistortImage(img1, intrinsic_matrixL, distCoeffL)

    # Secondly undistort images taken with right camera
    #print('Undistort the right images')
    undistorted_image_R = disp.UndistortImage(img2, intrinsic_matrixR, distCoeffR)

    # --> calculate disparity images
    disparity_visual = disp.getDisparity(imgLeft=undistorted_image_L, imgRight=undistorted_image_R, method="BM")

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

def isObsticleInFront(img):
    meanValue = meanPixelSUM(img)
    print "meanValue for disparity image"
    print meanValue
    if meanValue < 1.7:
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


def findBiggestObject(img, pts_que_center, radiusTresh=40):
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
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > radiusTresh: # works as a treshold
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(img, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(img, center, 5, (0, 0, 255), -1)

    # update the points queue
    pts_que_center.appendleft(center)

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

    return img, center


### Main method
def Main():

    pts_que_center = deque(maxlen=15)
    # tresh value
    pts_que_center_list = [deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)] # list holds 5 elements

    # tuning parameters
    radiusTresh = 40


    # parameters
    #b:= base offset, (the distance *between* your cameras)
    base_offset = 30.5

    #f:= focal length of camera,
    fx = 2222
    focal_length = (fx*35)/1360
    #Distance_map = (base_offset*focal_length)/disparity_visual




    with Vimba() as vimba:
        system = vimba.getSystem()
        system.runFeatureCommand("GeVDiscoveryAllOnce")
        #system1.runFeatureCommand("GeVDiscoveryAllOnce")


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
            c1.StreamBytesPerSecond = 50000000 #100000000   # gigabyte ethernet cable
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
            c2.StreamBytesPerSecond = 50000000 #100000000 --> taking the half og MAXIMUM = 124000000, and gives a margin
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
        folderName_saveImages = "savedImages"
        toktName = "tokt1"

        # Load camerea parameters
        # load calibration parameters
        [intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR] = disp.loadCameraParameters()



        while 1:
            # get status of camera 1
            #time.sleep(1)

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

            # Set the "shutter speed" --> How long the camerea takes in light to make the image
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

                    ################################################################
                    #proc
                    # Erode to remove noise
                    IMGbw_PrepcalcCentroid = proc.prepareDisparityImage_for_centroid(disparity_visual)

                    # calculate the centers of the small "objects"
                    image_withCentroids, centerCordinates = proc.findCentroids(IMGbw_PrepcalcCentroid)

                    #cv2.imshow("image after finding centroids", image_withCentroids)

                    centerCordinates = np.asarray(centerCordinates) # make list centerCordinates into numpy array

                    imageDraw, pixelSizeOfObject = proc.drawStuff(centerCordinates, disparity_visual.copy())

                    object_real_world_mm = 500 # 1000mm = 1 meter
                    distance_mm = proc.calcDistanceToKnownObject(object_real_world_mm, pixelSizeOfObject)
                    # calculate an estimate of distance

                    print "distance_mm"
                    print distance_mm

                    objectCenter = proc.getAverageCentroidPosition(centerCordinates)

                    ####### make image that buffers "old" centerpoints, and calculate center of the biggest centroid -- hopefully that is the biggest object
                    imgStaaker, center = findBiggestObject(disparity_visual.copy(), pts_que_center, radiusTresh=radiusTresh)

                     # update the points queue
                    # pts_que_center.appendleft(objectCenter)
                    #pts_que_center.appendleft(center)

                    ######

                    #cv2.imshow("image after finding minimum bounding rectangle of object", imageDraw )

                    #draw the new center in white
                    centerCircle_Color = (255, 255, 255)
                    cv2.circle(imgStaaker, objectCenter, 10, centerCircle_Color)

                    cv2.imshow("image staaker", imgStaaker )

                    #cv2.imshow('center object', disparity_visual)

                    # get direction of objects

                    Xpos = proc.findXposMessage(objectCenter)
                    print "Xpos"
                    print Xpos

                    XposCenterBiggestObject = proc.findXposMessage(center)
                    print "XposCenterBiggestObject"
                    print XposCenterBiggestObject

                    #######################################################################################

                    # "displaying disparity"
                    #cv2.imshow("disparityMovie", disparity_visual)
                    dispTime = (time.time() - start_time) + 0.0035

                    # Imporve disparity image, by using a scale sin(20) to sin(50) --> becouse the camera is tilted 35 or 45 degrees?
                    # make an array of values from sin(20) to sin(50)
                    disparity_visual_adjusted = camereaAngleAdjuster(disparity_visual)
                    cv2.imshow("disparity_visual_adjusted", disparity_visual_adjusted)


                    # apply mask so that disparityDisctance() don`t divide by zero
                    #disparity_visual = disparity_visual  #.astype(np.float32) / 16.0
                    #TODO: doobble check this--> i think i need it to not divide by zero
                    min_disp = 1  # 16
                    num_disp = 112-min_disp
                    #disparity_visual=(disparity_visual-min_disp)/num_disp
                    disparity_visual_adjusted = (disparity_visual_adjusted -min_disp)/num_disp


                    # calculate the Depth_map
                    Depth_map = disparityDisctance(disparity_visual, focal_length, base_offset)
                    #cv2.imshow("Depth_map", Depth_map)


                    # save the images that has been used to create disparity
                    pairNumber = pairNumber + 1
                    imgNameString_L = folderName_saveImages + "/" + toktName + "_L_" + str(pairNumber) + ".jpg"
                    imgNameString_R = folderName_saveImages + "/" + toktName + "_R_" + str(pairNumber) + ".jpg"

                    imgNameString_DISTANCE = folderName_saveImages + "/" + toktName + "_Depth_map_" + str(pairNumber) + ".jpg"

                    imgNameString_DISPARITY = folderName_saveImages + "/" + toktName + "_Disp_map_" + str(pairNumber) + ".jpg"

                    # writing the images to diske
                    cv2.imwrite(imgNameString_L, img1)
                    cv2.imwrite(imgNameString_R, img2)
                    cv2.imwrite(imgNameString_DISTANCE, Depth_map)
                    cv2.imwrite(imgNameString_DISTANCE, Depth_map)


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

                    # Compare the 3 parts, (Left, Center, Right) with each other to find in what area the object is.
                    #returnValue = compare3windows(depthMap, somthing )
                    # Image ROI
                    if isObsticleInFront(disparity_visual):
                        directionMessage = obstacleAvoidanceDirection(disparity_visual)
                    else:
                        directionMessage = "CONTINUE"


                    print "directionMessage"
                    print directionMessage


                    # TODO: Send UDP message to pc running labview program
                    sendUDPmessage(directionMessage)

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

# todo: filter out everythong further away than 2 meter to test