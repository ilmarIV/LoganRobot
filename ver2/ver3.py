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

    kernel_size = int(lines[3].split()[1])

KERNEL = np.ones((kernel_size, kernel_size), np.uint8)


def sendMainboard(ser, speeds_queue, throws_queue, stop_queue):
    while True:

        if not stop_queue.empty():
            return

        if not throws_queue.empty():
            throws = throws_queue.get()
            throw_msg = "d:" + str(throws) + "\r\n"
            ser.write(throw_msg.encode('UTF-8'))
            time.sleep(0.2)

            while ser.inWaiting() > 0:
                ser.read()

        if not speeds_queue.empty():
            speeds = speeds_queue.get()

            speed_msg = "sd:" + str(speeds[0]) + ":" + str(speeds[1]) + ":" + str(speeds[2]) + "\r\n"

            ser.write(speed_msg.encode('UTF-8'))
            time.sleep(0.02)

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

    thresholded_ball = cv2.inRange(hsv_image, BALL_LOWER, BALL_UPPER)
    thresholded_basket = cv2.inRange(hsv_image, BASKET_LOWER, BASKET_UPPER)

    morph_ball = cv2.morphologyEx(thresholded_ball, cv2.MORPH_CLOSE, KERNEL)
    morph_basket = cv2.morphologyEx(thresholded_basket, cv2.MORPH_CLOSE, KERNEL)

    return color_image, depth_frame, morph_ball, morph_basket


def getKeypoints(image, detector):
    keypoints = detector.detect(image)
    return keypoints


def findNearest(keypoints):
    nearest_x = keypoints[0].pt[0]
    nearest_y = keypoints[0].pt[1]
    nearest_size = keypoints[0].size

    for keypoint in keypoints:
        if keypoint.pt[1] > nearest_y:
            nearest_y = keypoint.pt[1]
            nearest_x = keypoint.pt[0]
            nearest_size = keypoint.size

    return nearest_x, nearest_y, nearest_size


def driveToBall(queue, ball_x, straight_speed, right_speed, left_speed):
    if abs(ball_x - 320) < 10:
        queue.put(straight_speed)

    else:
        if ball_x < 320:
            queue.put(left_speed)
        else:
            queue.put(right_speed)

    return


def turnRight(queue, turn_speed):
    queue.put(turn_speed)
    return


def turnAroundBallRight(queue, ball_x, turn_speed, compensate, sideways):
    #print("otsin korvi")
    if abs(ball_x - 320) < 5:
        queue.put(turn_speed)
    elif ball_x < 320:
        queue.put(sideways)
        #queue.put([-5, 20, -5])
    else:
        queue.put(compensate)
        #queue.put([10, 10, 10])

    return


def turnAroundBall(queue, ball_x, basket_x, right_turn, left_turn, compensate_right, compensate_left, sideways_right, sideways_left):
    if basket_x < 320:
       # print("vasakule")

        if abs(ball_x - 320) < 5:
            queue.put(left_turn)
        elif ball_x < 320:
            #queue.put([-10, -10, -10])
            queue.put(compensate_left)
        else:
            #queue.put([5, -20, 5])
            queue.put(sideways_left)
    else:
       # print("paremale")

        if abs(ball_x - 320) < 5:
            queue.put(right_turn)
        elif ball_x < 320:
            #queue.put([-5, 20, -5])
            queue.put(sideways_right)
        else:
            #queue.put([10, 10, 10])
            queue.put(compensate_right)

    return


def throw(throw_queue, speeds_queue, basket_dist):
    start = time.time()
    while time.time() - start <= 0.7:
        throw_queue.put(210)
        speeds_queue.put([25, 0, -25])

    while not throw_queue.empty():
        throws_queue.get()
    throws_queue.put(100)

    while not speeds_queue.empty():
        speeds_queue.get()
    speeds_queue.put([0, 0, 0])

    return


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
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

ser = Serial('/dev/ttyACM0', baudrate=115200, timeout=1)
speeds_queue = queue.Queue()
throws_queue = queue.Queue()

stop_queue = queue.Queue()

mbt = Thread(target=sendMainboard, name="mbt", args=(ser, speeds_queue, throws_queue, stop_queue))

have_ball = False

cv2.namedWindow("original")
cv2.namedWindow("ball")
cv2.namedWindow("basket")

ball_kp = None
basket_kp = None
ball_img = None
basket_img = None

mbt.start()

#slow driving speeds
forward_slow = [10, 0, -10]
right_slow = [20, 0, -10]
left_slow = [10, 0, -20]

#fast driving speeds
forward_fast = [50, 0, -50]
right_fast = [60, 0, -40]
left_fast = [40, 0, -60]

#turn right speed
right_turn = [20, 20, 20]

#tuen around ball
around_right_slow = [0, 20, 0]
around_right_fast = [0, 40, 0]

around_left_slow = [0, -20, 0]
around_left_fast = [0, -40, 0]

compensate_left_slow = [-5, -5, -5]
compensate_left_fast = [-15, -15, -15]
compensate_right_slow = [5, 5, 5]
compensate_right_fast = [15, 15, 15]

sideways_left_slow = [8, -25, 8]
sideways_left_fast = [10, -40, 10]
sideways_right_slow = [-8, 25, -8]
sideways_right_fast = [-10, 40, -10]


while True:

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

    frames = getFrames(pipeline)
    if frames:

        draw_image, depth_frame, ball_img, basket_img = getImages(frames)

        basket_kp = getKeypoints(basket_img, basket_detector)

        ball_kp = getKeypoints(ball_img, ball_detector)
        if ball_kp:
            ball_x, ball_y, ball_size = findNearest(ball_kp)

        if have_ball:

            if ball_kp and ball_y > 370:

                if basket_kp:
                    basket_x, basket_y = basket_kp[0].pt[0], basket_kp[0].pt[1]
                    basket_dist = depth_frame.get_distance(int(basket_x), int(basket_y))

                    if abs(basket_x - 320) < 5 and abs(ball_x - 320) < 5:
                        while not speeds_queue.empty():
                            speeds_queue.get()

                        throw(throws_queue, speeds_queue, basket_dist)
                        have_ball = False
                        #print(basket_x)

                    elif abs(basket_x - 320) < 80:
                        turnAroundBall(speeds_queue, ball_x, basket_x, around_right_slow, around_left_slow, compensate_right_slow, compensate_left_slow, sideways_right_slow, sideways_left_slow)

                    else:
                        turnAroundBall(speeds_queue, ball_x, basket_x, around_right_fast, around_left_fast, compensate_right_fast, compensate_left_fast, sideways_right_fast, sideways_left_fast)

                else:
                    turnAroundBallRight(speeds_queue, ball_x, around_right_fast, compensate_right_fast, sideways_right_fast)

            else:
                have_ball = False

        else:
            if ball_kp:

                if ball_y > 380 and ball_size > 40:
                    have_ball = True

                elif ball_y > 340:
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

cv2.destroyAllWindows()
pipeline.stop()
stop_queue.put(True)
mbt.join()
