# ROV_objectAvoidance_StereoVision

This program uses the stream from 2 stereo camereas using Vimba SDK to connect to the camereas
It uses the python wrapper to use the Vimba SDK

It is going to detect objects underwater, and the send a message to the control system that will plan a new path when objects are in it`s path. 


# Development folder
For testing of methods with stereoImages

testImages folder, contains image sets for experimenting with different methods.

to run program
python test_pair.py

# production folder
This code is produced to work with two camereas that it will stream from

to run program
python streamCam.py