import superpixel
import cv2

# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

import cProfile

# to train classifier
from localbinarypatterns import LocalBinaryPatterns

from sklearn.svm import LinearSVC
from skimage import feature
import numpy as np

# import the necessary packages
from imutils import paths

# to save and load, the model that is created from the classification
from sklearn.externals import joblib


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
        #centerCircle_Color = (0, 0, 0)
        #cv2.circle(imgBW, ctr, 4, centerCircle_Color)

        centerCordinates.append(ctr)
        #cv2.circle(imgBW, ctr, 5, (0,0,0))
        ctrTimes = 1
        #if ctrTimes == 1:  # Only wants the first ctr
        #    break

        #print "ctr"
        #print ctr
        #print(centerCordinates[0])

    return imgBW, centerCordinates

def findBiggestObject(img, radiusTresh, isObstacleInfront_based_on_radius):
    #width, height = img.shape[:2][::-1]

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

	'''
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
	'''

    return img, biggestObjectCenter, isObstacleInfront_based_on_radius

def predictHistofSegments(segments, image, model):

	imageROIList, centerList, predictionList, image, maskedImage = extractROIofSegmentandCenterList(image, segments, model)

	#image = removeOutliers(maskedImage)
	radiusTresh = 1
	isObstacleInfront_based_on_radius = False

	img, biggestObjectCenter, isObstacleInfront_based_on_radius = findBiggestObject(maskedImage, radiusTresh, isObstacleInfront_based_on_radius)

	if isObstacleInfront_based_on_radius:
		cv2.imshow("biggest object image", img)
		cv2.waitKey(0)


	showPredictionOutput(image, segments, predictionList, centerList)

'''
def showBiggestObject(img, biggestObjectCenter):
	# show the output of the prediction
	for (i, segVal) in enumerate(np.unique(segments)):
		CORD = centerList[i]
		if predictionList[i] == "other":
			colorFont = (255, 0, 0)
		else:
			colorFont = (0, 0, 255)

		cv2.putText(image, predictionList[i], CORD , cv2.FONT_HERSHEY_SIMPLEX,
					1.0, colorFont , 3)
		merkedImage =  mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments)

	cv2.imshow("biggest object image", img)
	cv2.waitKey(0)
'''

def showPredictionOutput(image, segments, predictionList, centerList):
	# show the output of the prediction
	for (i, segVal) in enumerate(np.unique(segments)):
		CORD = centerList[i]
		if predictionList[i] == "other":
			colorFont = (255, 0, 0)
		else:
			colorFont = (0, 0, 255)

		cv2.putText(image, predictionList[i], CORD , cv2.FONT_HERSHEY_SIMPLEX,
					1.0, colorFont , 3)
		merkedImage =  mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments)

	cv2.imshow("segmented image", merkedImage)
	cv2.waitKey(0)


def extractROIofSegmentandCenterList(image,segments, model):
	desc = LocalBinaryPatterns(24, 8)
	imageROIList = []
	centerList = []
	predictionList = []
	# create mask
	maskedImage = np.zeros(image.shape[:2],
					dtype="uint8")  # This mask has the same width and height a the original image and has a default value of 0 (black).

	# loop over the unique segment values
	for (i, segVal) in enumerate(np.unique(segments)):
		# construct a mask for the segment
		print "[x] inspecting segment %d" % (i)
		mask = np.zeros(image.shape[:2], dtype="uint8")
		mask[segments == segVal] = 255

		imageMasked = cv2.bitwise_and(image, image, mask=mask)

		grayImage = cv2.cvtColor(imageMasked, cv2.COLOR_BGR2GRAY)
		imageMasked, centerCordinates = superpixel.findCentroid(grayImage)
		centerCordinate = centerCordinates[0]
		centerList.append(centerCordinate)

		# calling the cv2.findContours on a treshold of the image
		contours0, hierarchy = cv2.findContours(mask , cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		moments = [cv2.moments(cnt) for cnt in contours0]

		# rounded the centroids to integer.
		centroids = [(int(round(m['m10'] / m['m00'])), int(round(m['m01'] / m['m00']))) for m in moments]

		print 'len(contours0)'
		print len(contours0)

		for ctr in contours0:
			# 2 compute bounding box of countour
			# box = cv2.minAreaRect(ctr)
			#(x, y, w, h) = cv2.minAreaRect(ctr)
			(x, y, w, h) = cv2.boundingRect(ctr)

			# 3 extract the rectangular ROI
			# Extract the ROI from the image and draw a bounding box
			# surrounding the MRZ
			imageROI = image[y:y + h, x:x + w].copy()

			# Mask the imageROI here according to prediction

			# 4 pass that into descriptor to obtain feature vector.

			grayImageROI = cv2.cvtColor(imageROI, cv2.COLOR_BGR2GRAY)
			hist = desc.describe(grayImageROI)

			prediction = model.predict(hist)[0]
			predictionList.append(prediction)

			# construct a mask for the segment

			if prediction == "other":
				maskedImage[y:y + h, x:x + w] = 255

			if prediction == "ocean":
				maskedImage[y:y + h, x:x + w] = 0

			imageROIList.append(grayImageROI)
			# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

			# put the image "back together"
			#image[y:y + h, x:x + w] = grayImageROI

	#grayImageROI = cv2.bitwise_and(grayImageROI, grayImageROI, mask=mask)
	#image = cv2.bitwise_and(image, image, mask=mask)

	#cv2.imshow('imageROI', image)
	#cv2.waitKey(0)
	cv2.imshow('mask', maskedImage)
	cv2.waitKey(0)





	return imageROIList, centerList, predictionList, image, maskedImage


def resizeImage(image):
	(h, w) = image.shape[:2]
	width = 1360
	# calculate the ratio of the width and construct the
	# dimensions
	r = width / float(w)
	dim = (width, int(h * r))
	inter = cv2.INTER_AREA
	resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
	return resized


def main():
	#createdModel = False
	createdModel = True

	if createdModel == True:
		model = superpixel.loadModel()

	elif createdModel == False:
		# load the image and apply SLIC and extract (approximately)
		# the supplied number of segments
		image = cv2.imread("tokt1_R_1037.jpg")
		image = resizeImage(image)
		segments = slic(img_as_float(image), n_segments=100, sigma=5)

		# Add a picture of not water, so that the Linear SVM can run, on more than one label, when it fits a model.
		# this model is later used to decide the classification

		#imageOther = cv2.imread("raptors.png")
		imageOther = cv2.imread("raptors.png")
		imageOther = resizeImage(imageOther)
		segmentsOther = slic(img_as_float(imageOther), n_segments=100, sigma=5)

		# display segments
		#showSLICoutput(image, segments)

		data, labels = superpixel.getHistofContoursOfSegments(segments, image, labelName="ocean")

		dataOther, labelsOther = superpixel.getHistofContoursOfSegments(segmentsOther, imageOther, labelName="other")

		print 'length of dataOther array and labelsOther array'
		print str(len(dataOther))  # 108
		print str(len(labelsOther))  # 108

		print 'length of data array and labels array before merging'
		print str(len(data))  # 108
		print str(len(labels))  # 108

		dataList = data + dataOther
		labelsList = labels + labelsOther

		print 'length of data array and labels array after merging'
		print str(len(dataList)) # 109
		print str(len(labelsList)) # 109

		# Train a Linear SVM on the data
		model = LinearSVC(C=100.0, random_state=42)
		model.fit(dataList, labelsList)   # TODO: get this error: ValueError: setting an array element with a sequence.

		superpixel.saveModel(model)
	#########################################################################################################################################
	# test the prediction of the model
	#image = cv2.imread("tokt1_R_267.jpg")
	image = cv2.imread("transpondertowerIMG/tokt1_L_473.jpg")


	image = resizeImage(image)
	# image = cv2.imread("tokt1_R_137.jpg")
	segments = slic(img_as_float(image), n_segments=100, sigma=5)

	predictHistofSegments(segments, image, model)


if __name__ == '__main__':
    cProfile.run('main()')

