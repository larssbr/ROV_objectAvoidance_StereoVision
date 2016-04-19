import cv2
import numpy as np
from matplotlib import pyplot as plt
import processesBWdisparityIMG as find
import disparityMapCalc as disp

def checkEpipolarLines(img1,img2):
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m, n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    # Now we have the list of best matches from both the images.
    # Let s find the Fundamental Matrix
    #pts1 = np.int32(pts1)
    #pts2 = np.int32(pts2)
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    print "pts1"
    print pts1
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]



    # Next we find the epilines.
    # Epilines corresponding to the points in first image is drawn on second image

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)


    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    #img5 = np.float(img5)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape[:2]  #img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


##################Point Cloud CODE################################

# this ones is color point cloud
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

ply_header = (
'''ply
format ascii 1.0
element vertex {vertex_count}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
''')

# this ones is for gray scale point cloud
ply_headerColor = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
end_header
'''

# this method is for gray scale point cloud. I just want to have the positions of points in 3d
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

    h, w = disparity_image.shape[:2]
    Q = np.float32([[1, 0, 0, w / 2],
                    [0, -1, 0, h / 2],
                    [0, 0, focal_length, 0],
                    [0, 0, 0, 1]])
    '''

    # This is replaced because my results were always bad. Estimates are
    # taken from the OpenCV samples.

    width, height = disparity_image.shape[:2]
    #focal_length = 0.8 * width
    Q = np.float32([[1, 0, 0, -0.5 * width],
                    [0, -1, 0, 0.5 * height],
                    [0, 0, 0, -focal_length],
                    [0, 0, 1, 0]])


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
    #focal_length = 0.8 * width
    Q = np.float32([[1, 0, 0, -Cx],
                [0, -1, 0,  Cy],
                [0, 0, 0, -focal_length],
                [0, 0, 1, 0]])

    '''
    points = cv2.reprojectImageTo3D(disparity_image, Q)

    # get distance to obeject here


    colors = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)

    """
        Initialize point cloud with given coordinates and associated colors.

        ``coordinates`` and ``colors`` should be numpy arrays of the same
        length, in which ``coordinates`` is made of three-dimensional point
        positions (X, Y, Z) and ``colors`` is made of three-dimensional spectral
        data, e.g. (R, G, B).
    """
    coordinate_points = points.reshape(-1, 3)
    color_points = colors.reshape(-1, 3)

    # filter out infinity points using a mask
    mask = coordinate_points[:, 2] > coordinate_points[:, 2].min()
    coordinate_points = coordinate_points[mask]
    color_points = color_points[mask]

    # save the points with color at each cordinate
    coordinates = np.hstack([coordinate_points, color_points])  # Adding color component

    ply_string = ply_header % dict(vert_num=len(coordinates))
    for coordinate in coordinates:
        ply_string += "%f " % coordinate[0]
        ply_string += "%f " % coordinate[1]
        ply_string += "%f " % coordinate[2]
        ply_string += "%d " % coordinate[3]
        ply_string += "%d " % coordinate[4]
        ply_string += "%d\n" % coordinate[5]

    print "saving pointCloud to .ply file as output_file given name"
    output_file = "set.ply"
    with open(output_file, 'w') as f:
        #f.write(ply_string)
        f.write(ply_string.format( vertex_count=len(coordinate_points)))
        np.savetxt(f, coordinates, '%f %f %f %d %d %d')

    #return result # ply_string

def getDisparity(gray_left, gray_right, method="BM"):

    print gray_left.shape
    c, r = gray_left.shape
    if method == "BM":
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
        sbm.SADWindowSize = 9 #9
        sbm.numberOfDisparities = 96
        sbm.preFilterCap = 63 #63
        sbm.minDisparity = -21 #-21
        sbm.uniquenessRatio = 7
        sbm.speckleWindowSize = 0
        sbm.speckleRange = 8
        sbm.disp12MaxDiff = 1
        sbm.fullDP = False


        disparity = sbm.compute(gray_left, gray_right)
        disparity_visual = cv2.normalize(disparity, alpha=0, beta=255, norm_type=cv2.cv.CV_MINMAX, dtype=cv2.cv.CV_8U)

    return disparity_visual

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


     # make undistort images grayscale
    undistorted_image_L = cv2.cvtColor(undistorted_image_L, cv2.COLOR_BGR2GRAY)
    undistorted_image_R = cv2.cvtColor(undistorted_image_R, cv2.COLOR_BGR2GRAY)

    # TODO: check if the undistortion is accaptable
    # checkEpipolarLines from: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
    #checkEpipolarLines(undistorted_image_L, undistorted_image_R) #--> disparity_visual does not work get data type error

    # --> calculate disparity images
    disparity_visual = getDisparity(gray_left=undistorted_image_L, gray_right=undistorted_image_R, method="BM")  #method="BM")

    return disparity_visual

def disparityDisctance(disparity_visual, focal_length, base_offset):
    #D:= Distance of point in real world,
    #b:= base offset, (the distance *between* your cameras)
    #f:= focal length of camera,
    #d:= disparity:

    #D = b*f/d
    Depth_map = (base_offset*focal_length)/disparity_visual
    return Depth_map

def update_disparity_map(dispIMG):
        """
        Update disparity map in GUI.

        The disparity image is normalized to the range 0-255 and then divided by
        255, because OpenCV multiplies it by 255 when displaying. This is
        because the pixels are stored as floating points.
        """
        #disparity = self.block_matcher.get_disparity(self.pair)
        norm_coeff = 255 / dispIMG.max()
        cv2.imshow("updated disparity image", dispIMG * norm_coeff / 255)
        cv2.waitKey()


def threeDPointsFromDisparity(row, col, Q, dispImg): # compute_3D_world_coordinates
    #cv::Mat_<cv::Vec3f> XYZ(disparity32F.rows, disparity32F.cols);   #// Output point cloud
    rows, cols = dispImg.shape[:2]
    #cv::Mat_<float> vec_tmp(4,1)
    vec_tmp = np.zeros([4,1])
    '''
    for rows in dispImg[1]:
        for cols in dispImg[2]:
            1+1
    for(int y=0; y<disparity32F.rows; ++y):
        for(int x=0; x<disparity32F.cols; ++x) :
            #vec_tmp(0)=x;
            vec_tmp[0] = x

            #vec_tmp(1)=y;
            vec_tmp[1] = y
            #vec_tmp(2)=disparity32F.at<float>(y,x);
            vec_tmp[2]=disparity32F.at<float>(y,x)
            #vec_tmp(3)=1
            vec_tmp[3] = 1

            vec_tmp = Q*vec_tmp

            #vec_tmp /= vec_tmp(3)
            vec_tmp = vec_tmp / vec_tmp[3]

            cv::Vec3f &point = XYZ.at<cv::Vec3f>(y,x)
            point[0] = vec_tmp(0)
            point[1] = vec_tmp(1)
            point[2] = vec_tmp(2)




    #Adding the code used to compute 3D coords from disparity:


    #cv::Vec3f *StereoFrame::compute_3D_world_coordinates(int row, int col,shared_ptr<StereoParameters> stereo_params_sptr){

    #cv::Mat Q_32F;

    #stereo_params_sptr->Q_sptr->convertTo(Q_32F,CV_32F);
    Q_32F = np.float32(Q)
    dispImg = np.float32(dispImg)

    #cv::Mat_<float> vec(4,1);
    vec = np.array(4,1)

    #vec(0) = col

    #vec(1) = row

    #vec(2) = this->disparity_sptr->at<float>(row,col)

    # Discard points with 0 disparity
    #if(vec(2)==0) return NULL;
    if(vec(2)==0):
        return NULL # todo: how do i dicard stuff in python?
    #vec(3)=1;
    #vec(3)=1

    #vec = Q_32F*vec;
    VEC = Q_32F*vec

    #vec /= vec(3);  # --> 	vec = vec / vec(3);
    vec = vec / vec(3)

    #Discard points that are too far from the camera, and thus are highly unreliable
    #if(abs(vec(0))>10 || abs(vec(1))>10 || abs(vec(2))>10) return NULL;
    if(np.abs(vec(0))>10 or np.abs(vec(1))>10 or np.abs(vec(2))>10):
        return NULL  # todo: how do i dicard stuff in python?

    #cv::Vec3f *point3f = new cv::Vec3f();
    point3f = np.array()
    #(*point3f)[0] = vec(0);
    point3f[0] = vec(0)
    #(*point3f)[1] = vec(1);
    point3f[1] = vec(1)
    #(*point3f)[2] = vec(2);
    point3f[2] = vec(2)

    #return point3f;
    return point3f
    '''

def mainProcess():

    filenameDisp = r"savedImages\tokt1_Depth_map_1.jpg"
    #dispIMG = cv2.imread(filenameDisp, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    filenameLeft = r"savedImages\tokt1_L_1.jpg"
    filenameRight = r"savedImages\tokt1_R_1.jpg"

    IMG_L = cv2.imread(filenameLeft)  #.astype(np.float32)/16.0
    IMG_R = cv2.imread(filenameRight)  #.astype(np.float32)/16.0

     # load calibration parameters
    [intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR] = disp.loadCameraParameters()

    dispIMG = disparityCalc(IMG_L, IMG_R, intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR)   #.astype(np.float32)/16.0

    #b:= base offset, (the distance *between* your cameras)
    base_offset = 30.5

    #f:= focal length of camera,
    fx = 2222
    focal_length = (fx*35)/1360
    #Distance_map = (base_offset*focal_length)/disparity_visual

    colorIMG = IMG_L
    dispIMG = np.float32(dispIMG) # convert to correct datatype to create pointcloud
    colorIMG = np.float32(colorIMG)
    point_cloud(dispIMG, colorIMG, focal_length)

    #
    #points3D = reconstructScene(disparityMap, stereoParams);



    #3D point --> M = D * inv(K) * [u; v; 1],
    #where (u, v) are the image coordinates of the point.

    #D = disparityDisctance(dispIMG, focal_length, base_offset) # calculate the Depth D



    ################The 3D points are being calculated as follows:##############

    #cv::Mat XYZ(disparity8U.size(),CV_32FC3);
    #reprojectImageTo3D(disparity8U, XYZ, Q, false, CV_32F);


    # Convert to meters and create a pointCloud object
    #points3D = points3D ./ 1000;
    #ptCloud = pointCloud(points3D, 'Color', frameLeftRect);

    # Create a streaming point cloud viewer
    #player3D = pcplayer([-3, 3], [-3, 3], [0, 8], 'VerticalAxis', 'y', 'VerticalAxisDir', 'down');

    # Visualize the point cloud
    #view(player3D, ptCloud);


if __name__ == '__main__':
    mainProcess()