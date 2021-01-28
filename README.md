# LoganRobot
To run the code you must fist of all download the repo.
Next you must install the following libraries: 
cv2, pyrealsense2, Serial

Rest of the libraries are installed by deafault in Python 3.7

Depending on the operating system, you must aslo install realsense-viewer. 
Linux - https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
Windows - https://dev.intelrealsense.com/docs/installing-intel-realsense-sdk-20-for-windows

Before running any python code, you must disalbe some features in the realsense-viewer application. 
Under the RGB camera module, you must disable "enable auto_white_balance" and "auto_exposure_priority"



GetValues.py is the first file you must run. From there you can set the cameras thresholding values wich are saved into a values.txt file. Close the program with "q". Do not force close to program, otherwise the values are not saved. 
Version3.py is the main file. All of the games logic is stored and executed in 1 single file. Upon running the program, it will ask for an input. 
Input 1 to select pink basket, input 2 to select the blue basket. From there you can see different windows wich will display tresholded ball picture, trehsolded basket picture and the original camera picture. Press "q" to exit the program.. Do not force close the program otherwise the Threads will not close propely. 
