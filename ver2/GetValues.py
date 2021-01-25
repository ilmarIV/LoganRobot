import numpy as np
from functools import partial
import cv2
import pyrealsense2 as rs

def GetValues():
    
    def updateValues(lst, index, value):
        lst[index] = value
        return
    
    choice = input("(0 - ball, 1 - pink basket, 2 - blue basket): ")
    with open("values.txt", "r") as file:
        lines = file.readlines()
        data = lines[int(choice)].split()
        if int(choice) == 0:
            kernel_size = int(lines[3].split()[1])
        else:
            kernel_size = int(lines[4].split()[1])
        
    lower_limits = np.asarray([int(x) for x in data[1:4]])
    upper_limits = np.asarray([int(x) for x in data[4:]])

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    cv2.namedWindow("original")
    cv2.namedWindow("hsv")
    cv2.namedWindow("processed")
    
    cv2.createTrackbar("h_low", "original", int(data[1]), 255, partial(updateValues, lower_limits, 0))
    cv2.createTrackbar("s_low", "original", int(data[2]), 255, partial(updateValues, lower_limits, 1))
    cv2.createTrackbar("v_low", "original", int(data[3]), 255, partial(updateValues, lower_limits, 2))
    cv2.createTrackbar("h_high", "original", int(data[4]), 255, partial(updateValues, upper_limits, 0))
    cv2.createTrackbar("s_high", "original", int(data[5]), 255, partial(updateValues, upper_limits, 1))
    cv2.createTrackbar("v_high", "original", int(data[6]), 255, partial(updateValues, upper_limits, 2))

    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    rgb_sensor = profile.get_device().first_color_sensor()
    rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
    rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
    rgb_sensor.set_option(rs.option.auto_exposure_priority, False)
    rgb_sensor.set_option(rs.option.brightness, 0)
    rgb_sensor.set_option(rs.option.contrast, 50)
    rgb_sensor.set_option(rs.option.exposure, 166)
    rgb_sensor.set_option(rs.option.gain, 64)
    rgb_sensor.set_option(rs.option.gamma, 300)
    rgb_sensor.set_option(rs.option.hue, 0)
    rgb_sensor.set_option(rs.option.saturation, 64)
    rgb_sensor.set_option(rs.option.sharpness, 50)
    rgb_sensor.set_option(rs.option.white_balance, 4600)

    while True:
        frames = pipeline.wait_for_frames()
        if frames:
            color_frame = frames.get_color_frame()
            color_array = np.asanyarray(color_frame.get_data())
            hsv = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV)
            thresholded = cv2.inRange(hsv, lower_limits, upper_limits)
            closing = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

            cv2.imshow("original", color_array)
            cv2.imshow("hsv", hsv)
            cv2.imshow("processed", closing)
        
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    with open("values.txt", "w") as file:
        lines[int(choice)] = data[0] + " " + str(lower_limits[0]) + " " + str(lower_limits[1]) + " " + str(lower_limits[2]) + " " + str(upper_limits[0]) + " " + str(upper_limits[1]) + " " + str(upper_limits[2]) + "\n"
        file.writelines(lines)
    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    GetValues()
