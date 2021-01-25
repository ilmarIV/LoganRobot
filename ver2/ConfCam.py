import numpy as np
from functools import partial
import cv2
import pyrealsense2 as rs

with open("cam_conf.txt", "r") as file:
    lines = file.readlines()
    brightness = int(lines[0].split()[0])
    contrast = int(lines[1].split()[0])
    exposre = int(lines[2].split()[0])
    gain = int(lines[3].split()[0])
    gamma = int(lines[4].split()[0])
    hue = int(lines[5].split()[0])
    saturation = int(lines[6].split()[0])
    sharpness = int(lines[7].split()[0])
    white_balance = int(lines[8].split()[0])

CHANGED_VALUES = [brightness, contrast, exposre, gain, gamma, hue, saturation, sharpness, white_balance]

def updateValues(index, diff, value):
    CHANGED_VALUES[index] = value + diff
    return

cv2.namedWindow("original")
cv2.namedWindow("hsv")

pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)

profile = pipeline.start(config)


rgb_sensor = profile.get_device().first_color_sensor()
rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
rgb_sensor.set_option(rs.option.auto_exposure_priority, False)

cv2.createTrackbar("brightness", "original", brightness + 64, 128, partial(updateValues, 0, -64))
cv2.createTrackbar("contrast", "original", contrast + 0, 100, partial(updateValues, 1, 0))
cv2.createTrackbar("exposre", "original", exposre - 1, 700, partial(updateValues, 2, 1))
cv2.createTrackbar("gain", "original", gain, 128 + 0, partial(updateValues, 3, 0))
cv2.createTrackbar("gamma", "original", gamma - 100, 400, partial(updateValues, 4, 100))
cv2.createTrackbar("hue", "original", hue + 180, 360, partial(updateValues, 5, -180))
cv2.createTrackbar("saturation", "original", saturation + 0, 100, partial(updateValues, 6, 0))
cv2.createTrackbar("sharpness", "original", sharpness + 0, 100, partial(updateValues, 7, 0))
cv2.createTrackbar("white_balance", "original", white_balance - 2500, 3700, partial(updateValues, 8, 2800))

while True:
    frames = pipeline.wait_for_frames()
    if frames:



        rgb_sensor.set_option(rs.option.brightness, CHANGED_VALUES[0])
        rgb_sensor.set_option(rs.option.contrast, CHANGED_VALUES[1])
        rgb_sensor.set_option(rs.option.exposure, CHANGED_VALUES[2])
        rgb_sensor.set_option(rs.option.gain, CHANGED_VALUES[3])
        rgb_sensor.set_option(rs.option.gamma, CHANGED_VALUES[4])
        rgb_sensor.set_option(rs.option.hue, CHANGED_VALUES[5])
        rgb_sensor.set_option(rs.option.saturation, CHANGED_VALUES[6])
        rgb_sensor.set_option(rs.option.sharpness, CHANGED_VALUES[7])
        rgb_sensor.set_option(rs.option.white_balance, CHANGED_VALUES[8])

        color_frame = frames.get_color_frame()
        color_array = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV)

        cv2.imshow("original", color_array)
        cv2.imshow("hsv", hsv)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

with open("cam_conf.txt", "w") as file:
    for i in range(9):
        lines[i] = str(CHANGED_VALUES[i]) + "\n"

    file.writelines(lines)

pipeline.stop()
cv2.destroyAllWindows()