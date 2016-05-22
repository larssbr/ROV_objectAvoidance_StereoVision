
import logging
import os
import cv2
import csv
from glob import glob
import re

# images class preps data structure for use by Stitch class
# facilitates being able to handle multiple types of image gathering methods (load from S3, etc.)
class Images:

    def __init__(self, dir_path):
        #self.logger = logging.getLogger()
        self.imageList = []
        #self.infolist = None
        self.image_width = 100
        self.imageHeight = 100
        self.filenames = []
        self.dir_path = dir_path

    # from main input argument is (args['dir']) --> dir_path = 'dir'
    def loadFromDirectory(self, dir_path=None, is_infofile=None):
        #self.logger.info("Searching for images and posAnd_dir.csv in: {}".format(dir_path))

        if self.dir_path == None:
            raise Exception("You need to input the directory string where source images are placed on your computer")
        if not os.path.isdir(self.dir_path):
            raise Exception("Directory string is not correct, check that your directory string is correct!")

        # grab pose data from csv # pose contains: filename,latitude,longitude,altitude,yaw,pitch,roll
        if is_infofile==True: # --> only grab infofile if there is one
            #self.logger.info("is_infofile was set to True")
            self.infolist = self.getPoseData(self.dir_path)
            if len(self.infolist) == 0:
                #self.logger.error("Error reading posAnd_dir.csv")
                return False

        # grab filenames of the images in the specified directory
        #self.filenames = self.getFilenames(self.dir_path)
        self.filenames = self.get_sorted_files2()

        if self.filenames == None: # --> if there is no files in directory, then STOP
            #self.logger.error("Error reading filenames, check if the directory is empty?")
            return False

        # ELSE do this
        # load the imagesz
        for i, img_filename in enumerate(self.filenames):
            #self.logger.info("Opening file: {}".format(img_filename))
            self.imageList.append(cv2.imread(img_filename))

        # set attributes of image_width & imageHeight for images (based on image 1),
        # -->  assumes all images are the same size
        (self.image_width, self.imageHeight) = self.getimage_shape(self.imageList[0])

        #self.logger.info("Data loaded successfully.")

    # Inbar Rose answer in thread http://stackoverflow.com/questions/11953824/how-to-read-filenames-in-a-folder-and-access-them-in-an-alphabetical-and-increas
    def get_sorted_files(self, dir_path):
        filenamelist = []
        for root, dirs, files in os.walk(dir_path):
            for name in files:
                fullname = os.path.join(root, name)
                filenamelist.append(fullname)
        return sorted(filenamelist)

    def getint(self, name):
        #basename = name.partition('.')
        basename = name.split('.')
        alpha, num = basename.split('_')
        #alpha, num = basename[0].split('_')
        return int(num)

    #tiffiles.sort(key=getint)


    ######## This part sorts the files _123 _1 _4 as _1 _4 _123. It is slow, but it works!!
    # Daniel DiPaolo answer in  http://stackoverflow.com/questions/4623446/how-do-you-sort-files-numerically
    def tryint(self, s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(self, s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [self.tryint(c) for c in re.split('([0-9]+)', s)]

    def sort_nicely(self, l):
        """ Sort the given list in the way that humans expect.
        """
        l.sort(key=self.alphanum_key)
        return l

    def get_sorted_files2(self):
        files_list = glob(os.path.join(self.dir_path, '*.jpg'))   # Todo, add support for .png and other filetypes
        return self.sort_nicely(files_list) #  files_list.sort(key=self.getint)  # sorted(files_list)

    ######## End of sorting function













    def getimage_shape(self, img):
        return img.shape[1], img.shape[0]

    def getimage_list(self):
        return self.imageList

    # getPoseData() is called within loadFromDirectory --> it is a helper method
    def getPoseData(self, dir_path):
        # load pose data
        #self.logger.info("Loading posAnd_dir.csv...")
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
        #self.logger.info("Read {} rows from posAnd_dir.csv".format(len(data)))
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
            #self.logger.info("Found {} files in directory: {}".format(len(filenames), sPath))   #Found 74 files in directory: datasets/example2
            return filenames

