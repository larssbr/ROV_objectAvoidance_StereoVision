zimport cv2
import numpy as np


###########
# HELPERS #
###########

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

########
# MAIN #
def main():
    # Open the left and right streams
    left_video = cv2.VideoCapture(1)
    right_video = cv2.VideoCapture(2)
        # 1 # must read in movie here
    filenameLeft = "myMovies/redballMovingUnderwater_LeftCam.mp4"
    filenameRight = "myMovies/redballMovingUnderwater_RightCam.mp4"


    imgLeft = "F:\disparityimg_testset\Run4\left1\LeftCameraRun4_211.png"
    imgRight = "F:\disparityimg_testset\Run4\right1\RightCameraRun4_211.png"

    #captureLeft = cv2.VideoCapture(filenameLeft)
    #captureRight = cv2.VideoCapture(filenameRight)

    left_video = cv2.VideoCapture()
    right_video = cv2.VideoCapture()

    left_video.open(filenameLeft)
    right_video.open(filenameRight)

    [intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR] = loadCameraParameters()

    ret_left, frame_left = left_video.read()
    ret_right, frame_right = right_video.read()

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

    cv2.destroyAllWindows()

    # Tuner GUI
    cv2.namedWindow('tuner')
    cv2.createTrackbar('minDisparity', 'tuner', tuner_minDisparity, 100, _nothing)
    cv2.createTrackbar('numDisparities', 'tuner', tuner_numDisparities, 2048, _nothing)
    cv2.createTrackbar('SADWindowSize', 'tuner', tuner_SADWindowSize, 19, _nothing)
    cv2.createTrackbar('P1', 'tuner', tuner_P1, 5000, _nothing)
    cv2.createTrackbar('P2', 'tuner', tuner_P2, 100000, _nothing)

    while ret_left is True and ret_right is True:
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


if __name__ == '__main__':
    main()