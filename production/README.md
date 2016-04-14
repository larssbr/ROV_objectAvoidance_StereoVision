# To run program

python streamCam.py

# Dependencies
 
 - socket
 - math
 - cProfile
 - time 
 - numpy
 - opencv 2.4.9
 - vimba 2 sdk innstaled on your computer
 - vimba driver innstaled on your etherent card
 - pymba --> python wrapper for vimba sdk must be in your project folder
 - set the vimbDLL path in ""pymba/vimbadll.py" to where the vimba sdk dll file is placed on your computer, in my case:
 
 vimbaC_path = r'C:\Program Files\Allied Vision\Vimba_2.0\VimbaC\Bin\Win64\VimbaC.dll'
 
 
 # streamCam.py is configured for Allied Vission Technology camera 1380c 
 
 # Information
 
 
 
 # Focal length
 
 focal_length = fx*(35/1360)
 
 focal_length = fy*(35/1024)
  
  
# TODO

- only get image per 0.3 second, to match the time it takes to create disparity image

