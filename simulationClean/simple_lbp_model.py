import cv2

# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

import cProfile

# to train classifier
#from localbinarypatterns import LocalBinaryPatterns

from sklearn.svm import LinearSVC
from skimage import feature
import numpy as np

# import the necessary packages
from imutils import paths

# to save and load, the model that is created from the classification
from sklearn.externals import joblib

from slicSuperpixel_lbp_method import LocalBinaryPatterns

class simpleModelTools:

    def __init__(self, createdModel, imageOcean, imageOther):
        self.imageOcean =  imageOcean
        self.imageOther =  imageOther

        #createdModel = True

        if createdModel == True:
            self.model = self.loadModel()

        elif createdModel == False:
            self.model = self.createModel()
            # need to load the model after it is created
            self.model = self.loadModel()


    def saveModel(self, model):
        joblib.dump(model, "model/Simple_model.pkl")

    def loadModel(self):
        return joblib.load("model/Simple_model.pkl")

    def createModel(self):
        dataClassOcean = simpleAnalyseImageTools(self.imageOcean, "ocean")


        data, labels = dataClassOcean.get_HistofImage("ocean")

        dataClassOther = simpleAnalyseImageTools(self.imageOther, "other")


        dataOther, labelsOther = dataClassOther.get_HistofImage("other")

        dataList = data + dataOther
        labelsList = labels + labelsOther

        # Train a Linear SVM on the data
        model = LinearSVC(C=100.0, random_state=42)
        model.fit(dataList, labelsList)

        self.saveModel(model)

    def get_model(self):
        return self.model

class simpleAnalyseImageTools:
    def __init__(self, image, labelName):
        self.image = image
        #self.segments = self.get_segments(image)
        #self.imageROIList = self.get_ROIofContoursList()
        self.labelName = labelName

    def get_HistofImage(self, labelName):
        # initialize the local binary patterns descriptor along with
        # the data and label lists
        desc = LocalBinaryPatterns(24, 8)  # numPoints = 24, radius = 8
        # initialize the image descriptor -- a 2D LAB histogram using A and B channel with 8 bins per channel
        #desc = LABHistogram([8, 8])
        data = []
        labels = []

        grayImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(grayImage)

        labels.append(labelName)
        data.append(hist)

        return data, labels

class simplePredictionTool:

    def __init__(self, image, model):

        #self.image = self.resizeImage(image)
        self.image = self.resizeImage(image)  # image
        self.model = model
        #self.radiusTresh = radiusTresh
        self.prediction = self.Predict_HistofImage()

        #self.centerCordinates = []
        #self.desc = LocalBinaryPatterns(24, 8)

        self.isObstacleInFront = self.isObstacleInfrontBasedOnPrediction()

        #self.segments = self.get_segments(image)
        #self.imageROIList = self.get_ROIofContoursList()


        #self.imageROIList, self.centerList, self.predictionList, self.maskedImage = self.extractROIofSegmentandCenterList()

    # self.data, self.labels = self.get_HistofContoursOfSegments()

    def resizeImage(self, image):
        (h, w) = image.shape[:2]
        width = 360
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
        inter = cv2.INTER_AREA
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized

    ######################################################################################

    def Predict_HistofImage(self):
        # initialize the local binary patterns descriptor along with
        # the data and label lists
        desc = LocalBinaryPatterns(24, 8)  # numPoints = 24, radius = 8
        #predictionList = []

        # for imageROI in self.imageROIList:
        # 4 pass that into descriptor to obtain feature vector.
        grayImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        hist = desc.describe(grayImage)

        prediction = self.model.predict(hist)[0]
        #predictionList.append(prediction)

        return prediction

    def isObstacleInfrontBasedOnPrediction(self):
        # show the output of the prediction

        if self.prediction == "other":
            #colorFont = (255, 0, 0)
            isObstacleInFront = True
        else:
            isObstacleInFront = False

        return isObstacleInFront

        #return isObstacleInFront

        #cv2.putText(image, predictionList[i], CORD, cv2.FONT_HERSHEY_SIMPLEX,1.0, colorFont, 3)
        #merkedImage = mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments)

    def get_isObstacleInFront(self):
        return self.isObstacleInFront