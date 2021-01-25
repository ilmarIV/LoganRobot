import cv2
import pyrealsense2 as rs
import numpy as np
import time
from serial import Serial
from threading import Thread
import queue

choice = int(input("(1 - pink basket, 2 - blue basket): "))

with open("values.txt", "r") as file:
    lines = file.readlines()

    BALL_LOWER = np.asarray([int(x) for x in lines[0].split()[1:4]])
    BALL_UPPER = np.asarray([int(x) for x in lines[0].split()[4:]])

    BASKET_LOWER = np.asarray([int(x) for x in lines[choice].split()[1:4]])
    BASKET_UPPER = np.asarray([int(x) for x in lines[choice].split()[4:]])

    ball_kernel_size = int(lines[3].split()[1])
    basket_kernel_size = int(lines[3].split()[1])

BALL_KERNEL = np.ones((ball_kernel_size, ball_kernel_size), np.uint8)
BASKET_KERNEL = np.ones((basket_kernel_size, basket_kernel_size), np.uint8)


def sendMainboard(ser, speeds_queue, stop_queue):
    while True:
        #time_start = time.time()

        if not stop_queue.empty():
            return

        if not speeds_queue.empty():
            speeds = speeds_queue.get()

            speed_msg = "sd:" + str(speeds[0]) + ":" + str(speeds[1]) + ":" + str(speeds[2]) + "\r\n"
            throw_msg = "d:" + str(speeds[3]) + "\r\n"

            ser.write(speed_msg.encode('UTF-8'))
            time.sleep(0.03)

            #while ser.inWaiting() > 0:
                #ser.read()

            ser.write(throw_msg.encode('UTF-8'))
            time.sleep(0.03)

            while ser.inWaiting() > 0:
                ser.read()


def getFrames(pipeline):
    frames = pipeline.wait_for_frames()

    return frames


def getImages(frames):
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    color_image = np.asanyarray(color_frame.get_data())
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    thresholded_ball = cv2.inRange(hsv_image, BALL_LOWER, BALL_UPPER)
    thresholded_basket = cv2.inRange(hsv_image, BASKET_LOWER, BASKET_UPPER)

    morph_ball = cv2.morphologyEx(thresholded_ball, cv2.MORPH_CLOSE, BALL_KERNEL)
    morph_basket = cv2.morphologyEx(thresholded_basket, cv2.MORPH_CLOSE, BASKET_KERNEL)

    return color_image, depth_frame, morph_ball, morph_basket, gray_image


def getKeypoints(image, detector):
    keypoints = detector.detect(image)

    return keypoints


def findNearest(keypoints, gray_img):
    nearest_x = 0
    nearest_y = 0
    nearest_size = 0

    for keypoint in keypoints:
        new_x = keypoint.pt[0]
        new_y = keypoint.pt[1]

        if new_y > nearest_y and checkBounds(gray_img, new_x, new_y):
            nearest_y = new_y
            nearest_x = new_x
            nearest_size = keypoint.size

    return nearest_x, nearest_y, nearest_size


def checkBounds(frame, ball_x, ball_y):
    #a = [120]
    if ball_y < 420:
        for y in range(int(ball_y) + 50, 479, 1):
            #a.append(frame[y][int(ball_x)])

            if frame[y][int(ball_x)] < 60:
                return False
    #print(max(a), min(a))
    return True


def reverse(queue ,speed):
    queue.put(speed)

    return


def driveToBall(queue, ball_x, straight_speed, right_speed, left_speed):
    if abs(ball_x - 424) < 10:
        queue.put(straight_speed)

    else:
        if ball_x < 424:
            queue.put(left_speed)
        else:
            queue.put(right_speed)

    return


def turnRight(queue, turn_speed):
    queue.put(turn_speed)

    return


def rotateAroundBallRight(queue, ball_x, right_turn, left_turn):
    if ball_x < 424:
        queue.put(left_turn)

    else:
        queue.put(right_turn)

    return


def rotateToBasket(queue, ball_x, basket_x, right_right, right_left, left_right, left_left):
    while not speeds_queue.empty():
        speeds_queue.get()

    if basket_x < 424:
        if ball_x < 424:
            queue.put(left_left)
        else:
            queue.put(left_right)

    else:
        if ball_x < 424:
            queue.put(right_left)
        else:
            queue.put(right_right)

    return


def turnAround(queue):
    print("turning")
    while not speeds_queue.empty():
        speeds_queue.get()

    queue.put([140, 140, 140, 900])

    return


def throw(speeds_queue, basket_dist, basket_x, ball_x):
    while not speeds_queue.empty():
        speeds_queue.get()

    start = time.time()
    while time.time() - start <= 1:
        speeds_queue.put([17, 0, -17, 1200])   #2m=1278    #1m=1190

    while not speeds_queue.empty():
        speeds_queue.get()

    return


blobparams_ball = cv2.SimpleBlobDetector_Params()
blobparams_ball.filterByArea = True
blobparams_ball.minArea = 3
blobparams_ball.maxArea = 10000
blobparams_ball.filterByCircularity = False
blobparams_ball.filterByConvexity = False
blobparams_ball.filterByColor = False
blobparams_ball.filterByInertia = False

blobparams_basket = cv2.SimpleBlobDetector_Params()
blobparams_basket.filterByArea = True
blobparams_basket.minArea = 100
blobparams_basket.maxArea = 1000000
blobparams_basket.filterByCircularity = False
blobparams_basket.filterByConvexity = False
blobparams_basket.filterByColor = False
blobparams_basket.filterByInertia = False

ball_detector = cv2.SimpleBlobDetector_create(blobparams_ball)
basket_detector = cv2.SimpleBlobDetector_create(blobparams_basket)

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

ser = Serial('/dev/ttyACM0', baudrate=115200, timeout=1)
speeds_queue = queue.LifoQueue()

stop_queue = queue.Queue()

mbt = Thread(target=sendMainboard, name="mbt", args=(ser, speeds_queue, stop_queue))

have_ball = False

cv2.namedWindow("original")
cv2.namedWindow("ball")
cv2.namedWindow("basket")
cv2.namedWindow("gray")

ball_kp = None
basket_kp = None
ball_img = None
basket_img = None

distances = [2, 2, 2, 2]

thrower_stop = 1000

#reverse speed
reverse_speed = [-20, 0, 20, thrower_stop]

#slow driving speeds
forward_slow = [10, 0, -10, thrower_stop]
right_slow = [20, 0, -10, thrower_stop]
left_slow = [10, 0, -20, thrower_stop]

#fast driving speeds
forward_fast = [60, 0, -60, thrower_stop]
right_fast = [80, 0, -50, thrower_stop]
left_fast = [50, 0, -80, thrower_stop]

#turn right speed
right_turn = [25, 25, 25, thrower_stop]

#turn around ball
around_w_basket_right_1_slow = [2, 7, 0, thrower_stop]
around_w_basket_left_1_slow = [2, 7, 0, thrower_stop]

around_w_basket_right_2_slow = [2, -7, 0, thrower_stop]
around_w_basket_left_2_slow = [0, -7, -2, thrower_stop]


around_w_basket_right_1 = [3, 19, 1, thrower_stop]
around_w_basket_left_1 = [-1, 19, -3, thrower_stop]

around_w_basket_right_2 = [3, -19, 1, thrower_stop]
around_w_basket_left_2 = [-1, -19, -3, thrower_stop]


around_wo_basket_right = [9, 50, 2, thrower_stop]
around_wo_basket_left = [-2, 50, -9, thrower_stop]

#around_wo_basket_right = [4, 19, 1, thrower_stop]
#around_wo_basket_left = [-1, 19, -4, thrower_stop]


mbt.start()

for i in range(2):
    ser.write(b"d:900\r\n")
    time.sleep(1)
    ser.write(b"d:1200\r\n")
    time.sleep(1)

#ser.write(b"fs:1\r\n")
while True:
    #time_start = time.time()

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

    frames = getFrames(pipeline)

    if frames:

        draw_image, depth_frame, ball_img, basket_img, gray_image = getImages(frames)

        basket_kp = getKeypoints(basket_img, basket_detector)
        if basket_kp:
            basket_x, basket_y = basket_kp[0].pt[0], basket_kp[0].pt[1]
            basket_dist = depth_frame.get_distance(int(basket_x), int(basket_y))

            """
            center_dist = depth_frame.get_distance(424, 100)
            distances = distances[1:] + [center_dist]
            distance = np.average(distances)

            if (distance < 0.5 and distance > 0.01) and basket_dist < 1:

                have_ball = False
                turnAround(speeds_queue)
            """

        ball_kp = getKeypoints(ball_img, ball_detector)
        if ball_kp:
            ball_x, ball_y, ball_size = findNearest(ball_kp, gray_image)

        if have_ball:

            if (ball_kp and ball_x > 0) and ball_y > 460:
                reverse(speeds_queue, reverse_speed)

            elif (ball_kp and ball_x > 0) and ball_y > 370:

                if basket_kp:
                    if basket_dist < 1.5 and (abs(basket_x - 424) < 7 and abs(ball_x - 424) < 3):
                            throw(speeds_queue, basket_dist, basket_x, ball_x)
                            have_ball = False

                    elif basket_dist >= 1.5 and (abs(basket_x - 424) < 3 and abs(ball_x - 424) < 3):
                            throw(speeds_queue, basket_dist, basket_x, ball_x)
                            have_ball = False

                    elif abs(basket_x - 424) < 10:
                        rotateToBasket(speeds_queue, ball_x, basket_x, around_w_basket_right_1_slow, around_w_basket_left_1_slow, around_w_basket_right_2_slow, around_w_basket_left_2_slow)

                    else:
                        rotateToBasket(speeds_queue, ball_x, basket_x, around_w_basket_right_1, around_w_basket_left_1, around_w_basket_right_2, around_w_basket_left_2)

                else:
                    rotateAroundBallRight(speeds_queue, ball_x, around_wo_basket_right, around_wo_basket_left)

            else:
                have_ball = False

        else:
            if ball_kp and ball_x > 0:

                if ball_y > 380:
                    have_ball = True

                elif ball_y > 330:
                    driveToBall(speeds_queue, ball_x, forward_slow, right_slow, left_slow)

                else:
                    driveToBall(speeds_queue, ball_x, forward_fast, right_fast, left_fast)

            else:
                turnRight(speeds_queue, right_turn)


        draw_image = cv2.drawKeypoints(draw_image, ball_kp, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        draw_image = cv2.drawKeypoints(draw_image, basket_kp, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow("original", draw_image)
        cv2.imshow("ball", ball_img)
        cv2.imshow("basket", basket_img)
        #cv2.imshow("gray", gray_image)

cv2.destroyAllWindows()
pipeline.stop()
stop_queue.put(True)
mbt.join()
