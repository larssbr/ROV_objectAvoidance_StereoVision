# labMethod.py

# import the necessary packages
from skimage import feature
import numpy as np
import cv2

class LABHistogram:

    def __init__(self, bins):
        #  self.bins is a list of three integers, designating the number of bins for each channel.
        self.bins = bins  # store the number of bins the histogram will use


    def describe(self, image):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns

        imageLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        #hist = cv2.calcHist([chans[1], chans[2]],channels= [1,2], mask=None, histSize=[0, 256, 0, 256])   #of 32 bins for each channel: [32, 32, 32].
        hist = cv2.calcHist([imageLAB], [1, 2], None, self.bins, [0, 256, 0, 256])
        hist = cv2.normalize(hist)
        hist_flatten = hist.flatten() # In order to more easily compute the distance between histograms, we simply flatten this histogram to have a shape of (N ** 3,)

        # return the histogram of Local Binary Patterns
        return hist_flatten

    def test(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # extract a 3D RGB color histogram from the image,
        # using 8 bins per channel, normalize, and update
        # the index
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist).flatten()
        #index[filename] = hist

        # METHOD #1: UTILIZING OPENCV
        # initialize OpenCV methods for histogram comparison
        OPENCV_METHODS = (
            ("Correlation", cv2.cv.CV_COMP_CORREL),
            ("Chi-Squared", cv2.cv.CV_COMP_CHISQR),
            ("Intersection", cv2.cv.CV_COMP_INTERSECT),
            ("Hellinger", cv2.cv.CV_COMP_BHATTACHARYYA))

        # loop over the comparison methods
        for (methodName, method) in OPENCV_METHODS:
            # initialize the results dictionary and the sort
            # direction
            results = {}
            reverse = False

            # if we are using the correlation or intersection
            # method, then sort the results in reverse order
            if methodName in ("Correlation", "Intersection"):
                reverse = True
        return 0