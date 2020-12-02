import cv2
import pyrealsense2 as rs
import numpy as np
import time
from serial import Serial
from threading import Thread

choice = int(input("(1 - pink basket, 2 - blue basket): "))

with open("values.txt", "r") as file:
    lines = file.readlines()

    ball_lower = np.asarray([int(x) for x in lines[0].split()[1:4]])
    ball_upper = np.asarray([int(x) for x in lines[0].split()[4:]])

    basket_lower = np.asarray([int(x) for x in lines[choice].split()[1:4]])
    basket_upper = np.asarray([int(x) for x in lines[choice].split()[4:]])

    kernel_size = int(lines[3].split()[1])

kernel = np.ones((kernel_size, kernel_size), np.uint8)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

blobparams_ball = cv2.SimpleBlobDetector_Params()
blobparams_ball.filterByArea = True
blobparams_ball.minArea = 5
blobparams_ball.maxArea = 10000
blobparams_ball.filterByCircularity = False
blobparams_ball.filterByConvexity = False
blobparams_ball.filterByColor = False
blobparams_ball.filterByInertia = False

blobparams_basket = cv2.SimpleBlobDetector_Params()
blobparams_basket.filterByArea = True
blobparams_basket.minArea = 10
blobparams_basket.maxArea = 10000
blobparams_basket.filterByCircularity = False
blobparams_basket.filterByConvexity = False
blobparams_basket.filterByColor = False
blobparams_basket.filterByInertia = False

ball_detector = cv2.SimpleBlobDetector_create(blobparams_ball)
basket_detector = cv2.SimpleBlobDetector_create(blobparams_basket)

turn_speed = 20
speed = 40

Ku = 1
Tu = 1

Kp = 0.6 * Ku
Ki = (1.2 * Ku) / Tu
Kd = (3 * Ku * Tu) / 40

seconds = time.time()
integral = 0
last_error = 0

ser = Serial('/dev/ttyACM0', baudrate=115200, timeout=1)

cv2.namedWindow("original")
cv2.namedWindow("ball_image")
cv2.namedWindow("basket_image")

have_ball = False
see_ball = False
color_image = None
ball_morph = None
basket_morph = None

running = True

def imageThread():
    global basket_x
    global basket_dist
    global ball_x
    global see_basket
    global see_ball
    global color_image
    global ball_morph
    global basket_morph

    while running:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if (color_frame and depth_frame):
            color_image = np.asanyarray(color_frame.get_data())

            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            ball_thresholded = cv2.inRange(hsv_image, ball_lower, ball_upper)
            basket_thresholded = cv2.inRange(hsv_image, basket_lower, basket_upper)

            ball_morph = cv2.morphologyEx(ball_thresholded, cv2.MORPH_CLOSE, kernel)
            basket_morph = cv2.morphologyEx(basket_thresholded, cv2.MORPH_CLOSE, kernel)

            ball_keypoints = ball_detector.detect(ball_morph)
            basket_keypoints = basket_detector.detect(basket_morph)

            color_image = cv2.drawKeypoints(color_image, ball_keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            color_image = cv2.drawKeypoints(color_image, basket_keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            see_ball = False
            if (ball_keypoints):
                see_ball = True
                nearest = ball_keypoints[0]
                nearest_y = int(nearest.pt[1])

                for ball_keypoint in ball_keypoints:
                    ball_y = int(ball_keypoint.pt[1])

                    if(ball_y > nearest_y):
                        nearest = ball_keypoint
                        nearest_y = int(nearest.pt[1])

                ball_x = int(nearest.pt[0])

            see_basket = False
            if(basket_keypoints):
                see_basket = True
                for basket_keypoint in basket_keypoints:
                    basket_x = int(basket_keypoint.pt[0])
                    basket_y = int(basket_keypoint.pt[1])

                    basket_dist = depth_frame.get_distance(basket_x, basket_y)

            cv2.imshow("original", color_image)
            cv2.imshow("ball_image", ball_morph)
            cv2.imshow("basket_image", basket_morph)

def mainboardMSG(speed_1, speed_2, speed_3):
    msg = "sd:" + str(speed_1) + ":" + str(speed_2) + ":" + str(speed_3) + "\r\n"
    ser.write(msg.encode('UTF-8'))
    while ser.inWaiting() > 0:
        ser.read()

def driveToBall():
    global seconds
    global integral
    global last_error

    dt = time.time() - seconds
    seconds = time.time()
    error = 320 - ball_x
    integral = integral + error * dt
    deriative = (error - last_error) / dt
    change = int(Kp * error + Ki * integral + Kd * deriative)
    last_error = error

    mainboardMSG(speed + change, 0, speed - change)


def turnRight():
    mainboardMSG(turn_speed, 0, -turn_speed)


def turnToBasket():
    if basket_x < 320:
        mainboardMSG(turn_speed, 0, -turn_speed)
    else:
        mainboardMSG(-turn_speed, 0, turn_speed)


def throw():
    print(basket_dist)

Thread(name="imageThread", target=imageThread).start()

print("f")

while running:
    if have_ball:
        if see_basket:
            if abs(basket_x - 320) < 4:
                throw()
            else:
                turnToBasket()
        else:
            turnRight()
    else:
        if see_ball:
            driveToBall()
        else:
            turnRight()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        imageThread().join()
        running = False

