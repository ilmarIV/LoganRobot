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


def sendMainboard(ser, speeds_queue, throws_queue):
    while True:
        if not throws_queue.empty():
            throws = throws_queue.get()
            throw_msg = "d:" + str(throws) + "\r\n"
            ser.write(throw_msg.encode('UTF-8'))
            time.sleep(0.2)
            #print("viskas###############################################")

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


def getImage(color_frame, lower, upper):
    color_image = np.asanyarray(color_frame.get_data())
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    thresholded_image = cv2.inRange(hsv_image, lower, upper)
    morph_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, KERNEL)
    return color_image, morph_image


def getBallImage(frames):
    color_frame = frames.get_color_frame()
    color_image, morph_image = getImage(color_frame, BALL_LOWER, BALL_UPPER)
    return color_image, morph_image


def getBasketImage(frames):
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    color_image, morph_image = getImage(color_frame, BASKET_LOWER, BASKET_UPPER)
    return depth_frame, color_image, morph_image


def getKeypoints(image, detector):
    return detector.detect(image)


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


def driveToBallSlow(queue, ball_x):
    if abs(ball_x - 320) < 20:
        queue.put([10, 0, -10])
        #queue.put([0, 0, 0])
    else:
        if ball_x < 320:
            queue.put([10, 0, -20])
            #queue.put([0, 0, 0])
        else:
            queue.put([20, 0, -10])
            #queue.put([0, 0, 0])
    return

def driveToBallFast(queue, ball_x):
    if abs(ball_x - 320) < 20:
        queue.put([30, 0, -30])
        #queue.put([0, 0, 0])
    else:
        if ball_x < 320:
            queue.put([30, 0, -45])
            #queue.put([0, 0, 0])
        else:
            queue.put([45, 0, -30])
            #queue.put([0, 0, 0])
    return


def turnToBasket(queue, basket_x):
    if basket_x < 320:
        queue.put([20, 20, 20])
        #queue.put([0, -5, 0])
    else:
        queue.put([-20, -20, -20])
        #queue.put([0, 5, 0])
    return


def turnRight(queue):
    queue.put([20, 20, 20])
    #queue.put([0, 0, 0])
    return

def turnAroundBallSlow(queue):
    #queue.put([5, 5, 5])
    queue.put([0, 15, 0])
    return

def turnAroundBallFast(queue):
    #queue.put([5, 5, 5])
    queue.put([0, 20, 0])
    return

def throw(throw_queue, speeds_queue, basket_dist):
    start = time.time()
    while time.time() - start <= 0.5:
        #print("viskamas")
        throw_queue.put(230)
        speeds_queue.put([35, 0, -35])

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

have_ball = False

ser = Serial('/dev/ttyACM0', baudrate=115200, timeout=1)
speeds_queue = queue.Queue()
throws_queue = queue.Queue()

mbt = Thread(target=sendMainboard, name="mbt", args=(ser, speeds_queue, throws_queue))

cv2.namedWindow("original")

ball_kp = None
basket_kp = None
ball_img = None
basket_img = None

mbt.start()


while True:

    frames = getFrames(pipeline)
    if frames:

        if have_ball:
            depth_frame, draw_image, basket_img = getBasketImage(frames)
            basket_kp = getKeypoints(basket_img, basket_detector)
            #print("leidsin palli")

            if basket_kp:
                basket_x, basket_y = basket_kp[0].pt[0], basket_kp[0].pt[1]
                basket_dist = depth_frame.get_distance(int(basket_x), int(basket_y))

                if abs(basket_x - 280) < 10:
                    throw(throws_queue, speeds_queue, basket_dist)
                    #print("uuele ringile")
                    have_ball = False

                elif abs(basket_x - 280) < 50:
                    turnAroundBallSlow(speeds_queue)
                   # print("keera korvini")

                else:
                    turnAroundBallFast(speeds_queue)
                   # print("keera korvini")

            else:
                turnAroundBallFast(speeds_queue)
                #print("paremale")

        else:
            draw_image, ball_img = getBallImage(frames)
            ball_kp = getKeypoints(ball_img, ball_detector)


            if ball_kp:
                ball_x, ball_y, ball_size = findNearest(ball_kp)

                if ball_y > 400 and ball_size > 40:
                    have_ball = True
                    #print("have ball läks true")

                elif ball_y > 350:
                    driveToBallSlow(speeds_queue, ball_x)
                    # print("sõida pallini")

                else:
                    driveToBallFast(speeds_queue, ball_x)
                   # print("sõida pallini")

            else:
                turnRight(speeds_queue)
               # print("paremale")

        draw_image = cv2.drawKeypoints(draw_image, ball_kp, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        draw_image = cv2.drawKeypoints(draw_image, basket_kp, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("original", draw_image)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cv2.destroyAllWindows()
pipeline.stop()
mbt.join()
