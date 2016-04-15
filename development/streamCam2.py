# This method runs one of the two cameras
from pymba import *
import numpy as np

import time
import sys
import cv2

with Vimba() as vimba:
    system = vimba.getSystem()

    system.runFeatureCommand("GeVDiscoveryAllOnce")
    time.sleep(0.2)

    camera_ids = vimba.getCameraIds()

    for cam_id in camera_ids:
        print("Camera found: ", cam_id)

    c0 = vimba.getCamera(camera_ids[0])
    c0.openCamera()


    try:
        #gigE camera
        print("Packet size:", c0.GevSCPSPacketSize)
        c0.StreamBytesPerSecond = 100000000
        print("BPS:", c0.StreamBytesPerSecond)
    except:
        #not a gigE camera
        pass

    #set pixel format
      # OPENCV DEFAULT
    time.sleep(0.2)

    frame = c0.getFrame()
    frame.announceFrame()

    c0.startCapture()

    framecount = 0
    droppedframes = []

    while 1:
        try:
            frame.queueFrameCapture()
            success = True
        except:
            droppedframes.append(framecount)
            success = False
        c0.runFeatureCommand("AcquisitionStart")
        c0.runFeatureCommand("AcquisitionStop")
        frame.waitFrameCapture(1000)
        frame_data = frame.getBufferByteData()
        if success:
            img = np.ndarray(buffer=frame_data,
                             dtype=np.uint8,
                             shape=(frame.height, frame.width, frame.pixel_bytes))
            cv2.imshow("test", img)
        framecount += 1
        k = cv2.waitKey(1)
        if k == 0x1b:
            cv2.destroyAllWindows()
            print("Frames displayed: %i" % framecount)
            print("Frames dropped: %s" % droppedframes)
            break


    c0.endCapture()
    c0.revokeAllFrames()

    c0.closeCamera()

    vimba.shutdown()

