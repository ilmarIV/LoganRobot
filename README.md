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

You can also test camera settings via code ConfCam.py which saves them in cam_conf.txt. It shows the hsv image which you are going to threshold later.

File test.py is for testign thrower speeds. You can give it speeds while code is running and write them down. This version is for test robot though.

GetValues.py is the first file you must run. From there you can set the cameras thresholding values wich are saved into a values.txt file. Close the program with "q". Do not force close to program, otherwise the values are not saved. 
Version3.py is the main file. All of the games logic is stored and executed in 1 single file. Upon running the program, it will ask for an input. 
Input 1 to select pink basket, input 2 to select the blue basket. From there you can see different windows wich will display tresholded ball picture, trehsolded basket picture and the original camera picture. Press "q" to exit the program.. Do not force close the program otherwise the Threads will not close propely. 

In the main logic we first start by looking for the ball. If we find the ball, we move to the ball, keep a certain distance from it and start to aligning the ball to the basket. If the baskets centre and the ball are aligned, we move forward and set the thrower speed according to the distance to the basket. Once the ball is throwed, we start the process again.
