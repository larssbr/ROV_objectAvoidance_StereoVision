#python colorMethods.py

'''
This script if with methods for adjusting colors, and to track colors

'''

import cv2
import numpy as np

class colorTool:

    def __init__(self, image):
        self.image = image
        # redBoundaries = [([17, 15, 100], [50, 56, 200])]
        # RED COLOR TRACKING BOUNDARIES
        self.lower = np.array([17, 15, 100])
        self.upper = np.array([50, 56, 200])


    def track(self):
        '''
        # define the list of boundaries
        boundaries = [
            ([17, 15, 100], [50, 56, 200]),  # red
            ([86, 31, 4], [220, 88, 50]),  # blue
            ([25, 146, 190], [62, 174, 250]),  # yellow
            ([103, 86, 65], [145, 133, 128])  # gray
        ]
        '''

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(self.image, self.lower, self.upper)
        output = cv2.bitwise_and(self.image, self.image, mask = mask)

        # show the images
        #cv2.imshow("images", np.hstack([image, output]))
        #cv2.waitKey(0)

        # Blur the mask
        bmask = cv2.GaussianBlur(mask, (5,5),0)

        # CENTROIDS
        # Take the moments to get the centroid
        moments = cv2.moments(bmask)
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
            cv2.circle(self.image, ctr, 4, centerCircle_Color)

        # Display full-color image
        WINDOW_NAME = "RedballsStalker"
        cv2.imshow(WINDOW_NAME, self.image)
        #cv2.waitKey(0)

        # Force image display, setting centroid to None on ESC key input
        if cv2.waitKey(1) & 0xFF == 27:
            ctr = None

    def claheAdjustImages(self):
        # --> This method does clahe on lab space to keep the color
        # transform to lab color space and conduct clahe filtering on l channel then merge and transform back

        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe = cv2.createCLAHE(clipLimit=6.0,tileGridSize=(8, 8))
        #self.logger.info("The clahe type {}".format(type(clahe.type)))
        print 'color adjusting image'
        lab_image = cv2.cvtColor(self.img, cv2.cv.CV_RGB2Lab)
        print 'converted image to Lab space'
        lab_planes = cv2.split(lab_image)
        lab_planes[0] = clahe.apply(lab_planes[0])
        print 'apply clahe to image channel 0 --> L sapce'
        # Merge the the color planes back into an Lab image
        lab_image = cv2.merge(lab_planes, lab_planes[0])
        print 'merge channels back and transform back to rgb space'
        return cv2.cvtColor(lab_image, cv2.cv.CV_Lab2RGB)



