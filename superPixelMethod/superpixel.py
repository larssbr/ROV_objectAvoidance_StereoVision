# USAGE
# python2.7 superpixel.py --image raptors.png

# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import cProfile


from collections import deque
from threading import Thread
#from queue import Queue
import time
import cv2
import imutils

# to train classifier
from localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from skimage import feature
import numpy as np

# import the necessary packages
from imutils import paths

# to save and load, the model that is created from the classification
from sklearn.externals import joblib

def testValidity(image):
	# loop over the testing images
	for imagePath in paths.list_images(args["testing"]):
		# load the image, convert it to grayscale, describe it,
		# and classify it
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		hist = desc.describe(gray)
		prediction = model.predict(hist)[0]

		# display the image and the prediction
		cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
					1.0, (0, 0, 255), 3)
		cv2.imshow("Image", image)
		cv2.waitKey(0)

def showSLICoutput(image, segments):
	# show the output of SLIC

	#fig = plt.figure("Superpixels")
	#ax = fig.add_subplot(1, 1, 1)
	#ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
	#plt.axis("off")

	#plt.show()
	merkedImage =  mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments)
	cv2.imshow("segmented image", merkedImage)
	cv2.waitKey(0)

def getHistofSegments(segments, image, labelName):
	# initialize the local binary patterns descriptor along with
	# the data and label lists
	desc = LocalBinaryPatterns(24, 8)    # numPoints = 24, radius = 8
	data = []
	labels = []

	# loop over the unique segment values
	for (i, segVal) in enumerate(np.unique(segments)):
		# construct a mask for the segment
		print "[x] inspecting segment %d" % (i)
		mask = np.zeros(image.shape[:2], dtype="uint8")
		mask[segments == segVal] = 255

		imageROI = cv2.bitwise_and(image, image, mask=mask)



		grayImage = cv2.cvtColor(imageROI, cv2.COLOR_BGR2GRAY)
		hist = desc.describe(grayImage)

		# extract the label from the image path, then update the
		# label and data lists
		# labels.append(imageROI.split("/")[-2])
		labels.append(labelName)
		data.append(hist)

	return data, labels

def getHistofContoursOfSegments(segments, image, labelName):
	# initialize the local binary patterns descriptor along with
	# the data and label lists
	desc = LocalBinaryPatterns(24, 8)    # numPoints = 24, radius = 8
	data = []
	labels = []

	imageROIList = extractROIofSegment(image, segments)

	# 1. Loop over each superpixel and extract its contour.
	# 2. Compute bounding box of contour.
	# 3. Extract the rectangular ROI.

	# 4. Pass that into your descriptor to obtain your feature vector.
	# describeROI()

	for imageROI in imageROIList:

		#cv2.imshow('imageROI', imageROI)
		#cv2.waitKey(0)

		# 4 pass that into descriptor to obtain feature vector.

		grayImage = cv2.cvtColor(imageROI, cv2.COLOR_BGR2GRAY)
		hist = desc.describe(grayImage)


		# extract the label from the image path, then update the
		# label and data lists
		# labels.append(imageROI.split("/")[-2])
		labels.append(labelName)
		data.append(hist)

	return data, labels


def centroidPrediction(imgBW, segments, model):
	desc = LocalBinaryPatterns(24, 8)  # numPoints = 24, radius = 8
	imgBWCopy = imgBW.astype(np.uint8)

	h, w = imgBW.shape[:2]
	#contours0, hierarchy = cv2.findContours( imgBW.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	contours0, hierarchy = cv2.findContours( imgBWCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	moments = [cv2.moments(cnt) for cnt in contours0]

	# rounded the centroids to integer.
	centroids = [( int(round(m['m10']/m['m00'])), int(round(m['m01']/m['m00'])) ) for m in moments]

	predictionList = []
	centerList = []
	centerCordinates = []
	for ctr in centroids:
		centerCordinates.append(ctr)
		# draw a black little empty circle in the centroid position

		####### end -->Get centroid of imageROI, to write prediction at this position
		hist = desc.describe(ctr)
		# what if i try to predict on a centroid instead of the masked image?

		prediction = model.predict(hist)[0]

		predictionList.append(prediction)



	# loop over the testing images
	for (i, segVal) in enumerate(np.unique(segments)):
		# load the image, convert it to grayscale, describe it,
		# and classify it
		# construct a mask for the segment
		print "[x] inspecting segment %d" % (i)
		mask = np.zeros(image.shape[:2], dtype="uint8")
		mask[segments == segVal] = 255

		imageROI = cv2.bitwise_and(image, image, mask=mask)
		####### Get centroid of imageROI, to write prediction at this position

		grayImage = cv2.cvtColor(imageROI, cv2.COLOR_BGR2GRAY)

		grayImage, centerCordinates = findCentroid(grayImage)
		centerCordinate = centerCordinates[0]
		centerList.append(centerCordinate)

	showPredictionOutput(image, segments, predictionList, centerList)

	return imgBW, centerCordinates

def findCentroid(imgBW):

	imgBWCopy = imgBW.astype(np.uint8)

	h, w = imgBW.shape[:2]
	#contours0, hierarchy = cv2.findContours( imgBW.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	contours0, hierarchy = cv2.findContours( imgBWCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	moments = [cv2.moments(cnt) for cnt in contours0]

	# rounded the centroids to integer.
	centroids = [( int(round(m['m10']/m['m00'])), int(round(m['m01']/m['m00'])) ) for m in moments]

	centerCordinates = []
	for ctr in centroids:
		# draw a black little empty circle in the centroid position
		centerCircle_Color = (0, 0, 0)
		cv2.circle(imgBW, ctr, 4, centerCircle_Color)

		centerCordinates.append(ctr)
		#cv2.circle(imgBW, ctr, 5, (0,0,0))
		ctrTimes = 1
		#if ctrTimes == 1:  # Only wants the first ctr
		#    break

		#print "ctr"
		#print ctr
		#print(centerCordinates[0])

	return imgBW, centerCordinates
'''
def predictHistofSegments(segments, image, model):
	# Use the Linear SVM  model to classify subsequent texture images:

	# initialize the local binary patterns descriptor along with
	# the data and label lists
	desc = LocalBinaryPatterns(24, 8)  # numPoints = 24, radius = 8
	data = []
	labels = []
	predictionList = []
	centerList = []

	# loop over the testing images
	for (i, segVal) in enumerate(np.unique(segments)):
		# load the image, convert it to grayscale, describe it,
		# and classify it
		# construct a mask for the segment
		print "[x] inspecting segment %d" % (i)
		mask = np.zeros(image.shape[:2], dtype="uint8")
		mask[segments == segVal] = 255

		imageROI = cv2.bitwise_and(image, image, mask=mask)
		####### Get centroid of imageROI, to write prediction at this position

		grayImage = cv2.cvtColor(imageROI, cv2.COLOR_BGR2GRAY)

		grayImage, centerCordinates = findCentroid(grayImage)
		centerCordinate = centerCordinates[0]
		centerList.append(centerCordinate)

		####### end -->Get centroid of imageROI, to write prediction at this position
		hist = desc.describe(grayImage)
		# what if i try to predict on a centroid instead of the masked image?

		prediction = model.predict(hist)[0]

		predictionList.append(prediction)


		# display the image and the prediction
		#cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		#			1.0, (0, 0, 255), 3)
		#cv2.imshow("Image", image)
		#cv2.waitKey(0)

	showPredictionOutput(image, segments, predictionList, centerList)
'''
def predictHistofSegments(segments, image, model):
	desc = LocalBinaryPatterns(24, 8)  # numPoints = 24, radius = 8
	predictionList = []

	imageROIList, centerList = extractROIofSegmentandCenterList(image, segments)

	# 4. Pass that into your descriptor to obtain your feature vector.
	# describeROI()

	for imageROI in imageROIList:
		#cv2.imshow('imageROI', imageROI)
		#cv2.waitKey(0)

		# 4 pass that into descriptor to obtain feature vector.

		grayImage = cv2.cvtColor(imageROI, cv2.COLOR_BGR2GRAY)
		hist = desc.describe(grayImage)

		prediction = model.predict(hist)[0]

		predictionList.append(prediction)

	showPredictionOutput(image, segments, predictionList, centerList)


def showPredictionOutput(image, segments, predictionList, centerList):
	# show the output of the prediction
	for (i, segVal) in enumerate(np.unique(segments)):
		CORD = centerList[i]
		cv2.putText(image, predictionList[i], CORD , cv2.FONT_HERSHEY_SIMPLEX,
					1.0, (0, 0, 255), 3)
		merkedImage =  mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments)

	cv2.imshow("segmented image", merkedImage)
	cv2.waitKey(0)

def extractROIofSegment(image,segments):
	imageROIList = []
	# loop over the unique segment values
	for (i, segVal) in enumerate(np.unique(segments)):
		# construct a mask for the segment
		print "[x] inspecting segment %d" % (i)
		mask = np.zeros(image.shape[:2], dtype="uint8")
		mask[segments == segVal] = 255

		imageMasked = cv2.bitwise_and(image, image, mask=mask)

		#cv2.imshow('mask', mask)
		#cv2.imshow('imageMasked', imageMasked)
		#cv2.waitKey(0)

		# calling the cv2.findContours on a treshold of the image
		contours0, hierarchy = cv2.findContours(mask , cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		moments = [cv2.moments(cnt) for cnt in contours0]

		# rounded the centroids to integer.
		#centroids = [(int(round(m['m10'] / m['m00'])), int(round(m['m01'] / m['m00']))) for m in moments]

		print 'len(contours0)'
		print len(contours0)

		for ctr in contours0:
			# 2 compute bounding box of countour
			# box = cv2.minAreaRect(ctr)
			#(x, y, w, h) = cv2.minAreaRect(ctr)
			(x, y, w, h) = cv2.boundingRect(ctr)

			# 3 extract the rectangular ROI
			# extract the ROI from the image and draw a bounding box
			# surrounding the MRZ
			imageROI = image[y:y + h, x:x + w].copy()
			imageROIList.append(imageROI)
			# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	return imageROIList

def extractROIofSegmentandCenterList(image,segments):
	imageROIList = []
	centerList = []
	# loop over the unique segment values
	for (i, segVal) in enumerate(np.unique(segments)):
		# construct a mask for the segment
		print "[x] inspecting segment %d" % (i)
		mask = np.zeros(image.shape[:2], dtype="uint8")
		mask[segments == segVal] = 255

		imageMasked = cv2.bitwise_and(image, image, mask=mask)

		grayImage = cv2.cvtColor(imageMasked, cv2.COLOR_BGR2GRAY)
		imageMasked, centerCordinates = findCentroid(grayImage)
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
			# extract the ROI from the image and draw a bounding box
			# surrounding the MRZ
			imageROI = image[y:y + h, x:x + w].copy()
			imageROIList.append(imageROI)
			# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	return imageROIList, centerList

def saveModel(model):
	joblib.dump(model, "model/filename_model.pkl")
	# Save the created model
	# import pickle
	# savedModel = pickle.dumps(model)

def loadModel():
	return joblib.load("model/filename_model.pkl")

def main():
	createdModel = True

	if createdModel == True:
		model = loadModel()

	elif createdModel == False:
		# load the image and apply SLIC and extract (approximately)
		# the supplied number of segments
		image = cv2.imread("tokt1_R_1037.jpg")
		segments = slic(img_as_float(image), n_segments=100, sigma=5)

		# Add a picture of not water, so that the Linear SVM can run, on more than one label, when it fits a model.
		# this model is later used to decide the classification

		imageOther = cv2.imread("raptors.png")
		segmentsOther = slic(img_as_float(imageOther), n_segments=100, sigma=5)

		# display segments
		#showSLICoutput(image, segments)

		#data, labels = getHistofSegments(segments, image, labelName = "ocean")

		#dataOther, labelsOther = getHistofSegments(segments, image, labelName="other")

		data, labels = getHistofContoursOfSegments(segments, image, labelName="ocean")

		dataOther, labelsOther = getHistofContoursOfSegments(segmentsOther, image, labelName="other")

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

		saveModel(model)

	# test the prediction of the model

	image = cv2.imread("tokt1_R_267.jpg")
	#image = cv2.imread("tokt1_R_137.jpg")
	segments = slic(img_as_float(image), n_segments=100, sigma=5)
	predictHistofSegments(segments, image, model)


		# show the masked region
		#cv2.imshow("Mask", mask)
		#cv2.imshow("Applied", cv2.bitwise_and(image, image, mask=mask))
		#cv2.waitKey(0)


	'''
	# loop over the training images
	for imageROI in segments:
		# load the image, convert it to grayscale, and describe it
		#image = cv2.imread(imageROI)
		grayImage = cv2.cvtColor(imageROI, cv2.COLOR_BGR2GRAY)
		hist = desc.describe(grayImage)

		# extract the label from the image path, then update the
		# label and data lists
		#labels.append(imageROI.split("/")[-2])
		labels.append("ocean")
		data.append(hist)
	'''

	# graph representations across regions of the image

if __name__ == '__main__':
    cProfile.run('main()')


	# http://peekaboo-vision.blogspot.fr/2012/09/segmentation-algorithms-in-scikits-image.html
