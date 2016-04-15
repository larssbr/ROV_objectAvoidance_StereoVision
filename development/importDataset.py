import logging
import os
import cv2
import csv

# images class preps data structure for use by Stitch class
# facilitates being able to handle multiple types of image gathering methods (load from S3, etc.)
class Images:

    def __init__(self):
        self.logger = logging.getLogger()
        self.imageList = []
        #self.infolist = None
        self.image_width = 100
        self.imageHeight = 100
        self.filenames = []

    # from main input argument is (args['dir']) --> dir_path = 'dir'
    def loadFromDirectory(self, dir_path=None, is_infofile=None):
        self.logger.info("Searching for images and posAnd_dir.csv in: {}".format(dir_path))

        if dir_path == None:
            raise Exception("You need to input the directory string where source images are placed on your computer")
        if not os.path.isdir(dir_path):
            raise Exception("Directory string is not correct, check that your directory string is correct!")

        # grab pose data from csv # pose contains: filename,latitude,longitude,altitude,yaw,pitch,roll
        if is_infofile==True: # --> only grab infofile if there is one
            self.logger.info("is_infofile was set to True")
            self.infolist = self.getPoseData(dir_path)
            if len(self.infolist) == 0:
                self.logger.error("Error reading posAnd_dir.csv")
                return False

        # grab filenames of the images in the specified directory
        self.filenames = self.getFilenames(dir_path)
        if self.filenames == None: # --> if there is no files in directory, then STOP
            self.logger.error("Error reading filenames, check if the directory is empty?")
            return False

        # ELSE do this
        # load the imagesz
        for i, img_filename in enumerate(self.filenames):
            self.logger.info("Opening file: {}".format(img_filename))
            self.imageList.append(cv2.imread(img_filename))

        # set attributes of image_width & imageHeight for images (based on image 1),
        # -->  assumes all images are the same size
        (self.image_width, self.imageHeight) = self.getimage_shape(self.imageList[0])

        self.logger.info("Data loaded successfully.")

    def getimage_shape(self, img):
        return img.shape[1], img.shape[0]

    # getPoseData() is called within loadFromDirectory --> it is a helper method
    def getPoseData(self, dir_path):
        # load pose data
        self.logger.info("Loading posAnd_dir.csv...")
        reader = csv.DictReader(open(dir_path+'/posAnd_dir.csv')) # The name has to be: '/posAnd_dir.csv'
        data = []
        for row in reader:
            for key, val in row.iteritems():
                val = val.replace('\xc2\xad', '') # understand this \xc2\xad
                try:
                    row[key] = float(val)
                except ValueError:
                    row[key] = val
            data.append(row)
        self.logger.info("Read {} rows from posAnd_dir.csv".format(len(data)))
        # helper dict for quickly finding pose data in O(1)
        poseByFilenameList = dict((d["filename"],
                                   dict(d, index=i)) for (i, d) in enumerate(data))
        return poseByFilenameList

    # getFilenames() is called within loadFromDirectory --> it is a helper method
    def getFilenames(self, sPath):
        filenames = []
        for sChild in os.listdir(sPath):
            # check for valid file types here
            if os.path.splitext(sChild)[1][1:] in ['jpg', 'png']:
                sChildPath = os.path.join(sPath,sChild)
                filenames.append(sChildPath)
        if len(filenames) == 0:
            return None
        else:
            self.logger.info("Found {} files in directory: {}".format(len(filenames), sPath))   #Found 74 files in directory: datasets/example2
            return filenames