# python disparityMapCalc.py


import cv2
import numpy as np
from colorMethods import claheAdjustImages

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





    return [intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR]

def UndistortImage(image, intrinsic_matrix, distCoeff):
    # 1 Undistort the Image
    undistorted_image = cv2.undistort(image, intrinsic_matrix, distCoeff, None)

    # 2 return the undistorted image
    #undistorted_imgList.append(undistorted_img)
    return undistorted_image

def getDisparity(imgLeft, imgRight, method="BM"):

    gray_left = cv2.cvtColor(imgLeft, cv2.cv.CV_BGR2GRAY)
    gray_right = cv2.cvtColor(imgRight, cv2.cv.CV_BGR2GRAY)
    print gray_left.shape
    c, r = gray_left.shape
    if method == "BM":
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
        disparity = cv2.cv.CreateMat(c, r, cv2.cv.CV_32F)
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



        gray_left = cv2.cv.fromarray(gray_left)
        gray_right = cv2.cv.fromarray(gray_right)

        cv2.cv.FindStereoCorrespondenceBM(gray_left, gray_right, disparity, sbm)
        disparity_visual = cv2.cv.CreateMat(c, r, cv2.cv.CV_8U)
        cv2.cv.Normalize(disparity, disparity_visual, 0, 255, cv2.cv.CV_MINMAX)
        disparity_visual = np.array(disparity_visual)

    elif method == "SGBM":
        sbm = cv2.StereoSGBM()
        sbm.SADWindowSize = 9
        sbm.numberOfDisparities = 96
        sbm.preFilterCap = 63
        sbm.minDisparity = -21
        sbm.uniquenessRatio = 7
        sbm.speckleWindowSize = 0
        sbm.speckleRange = 8
        sbm.disp12MaxDiff = 1
        sbm.fullDP = False

        disparity = sbm.compute(gray_left, gray_right)
        disparity_visual = cv2.normalize(disparity, alpha=0, beta=255, norm_type=cv2.cv.CV_MINMAX, dtype=cv2.cv.CV_8U)

    return disparity_visual

def main():
# python disparityMapCalc.py

    #capture = cv2.VideoCapture(0)

    # Rectify images
    redballFrameFound_counter = 0


    # 1 # must read in movie here
    filenameLeft = "myMovies/redballMovingUnderwater_LeftCam.mp4"
    filenameRight = "myMovies/redballMovingUnderwater_RightCam.mp4"
    #captureLeft = cv2.VideoCapture(filenameLeft)
    #captureRight = cv2.VideoCapture(filenameRight)

    captureLeft = cv2.VideoCapture()
    captureRight = cv2.VideoCapture()

    captureLeft.open(filenameLeft)
    captureRight.open(filenameRight)

    # 2 # LOAD camera parameters
    print('Loading camera parameters')
    [intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR] = loadCameraParameters()

    #while True:
    while True:
        #print 'reads in movie'
        okay, imageLeft = captureLeft.read()
        okay, imageRight = captureRight.read()

        #type = imageLeft.type()
        #depth = imageLeft.depth()
        #print depth
        #print type

        print '++++++++++++++++++++++++++++++++++++++++++++'
        # Coloradjust images
        fixed_image_left = claheAdjustImages(imageLeft)
        fixed_image_right = claheAdjustImages(imageRight)
        print '============================================'

        # 3 Undistort images
        print('Step 3:  and rectify the original images')
        print('we use the intrinsic_matrix & distCoeff to undistort the images.')
        print(' ')
        # First undistort images taken with left camera
        print('Undistort the left images')
        undistorted_image_L = UndistortImage(imageLeft, intrinsic_matrixL, distCoeffL)

        # Secondly undistort images taken with right camera
        print('Undistort the right images')
        undistorted_image_R = UndistortImage(imageRight, intrinsic_matrixR, distCoeffR)


        # 4 Compute the disparity map
        print 'Compute the disparity map'
        disparityMovie = getDisparity(undistorted_image_L, undistorted_image_R, method="BM")
        cv2.imshow("disparityMovie", disparityMovie)
        cv2.waitKey(0)

        print '------------------------------------'
        print 'Calculate ply'

if __name__ == '__main__':
    main()



