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


def sendMainboard(ser, freq, speeds_queue):
    seconds = time.time()
    while True:
        if not speeds_queue.empty():
            speeds = speeds_queue.get()

            if time.time() >= seconds + freq:
                seconds = time.time()

                msg = "sd:" + str(speeds[0]) + ":" + str(speeds[1]) + ":" + str(speeds[2]) + "\r\n"
                ser.write(msg.encode('UTF-8'))

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
            nearest_size = keypoint.pt[2]

    return nearest_x, nearest_y, nearest_size


def driveToBall(queue, ball_x):
    if abs(ball_x - 320) < 10:
        queue.put([20, -20, 0])
    else:
        if ball_x < 320:
            queue.put([10, 30, 0])
        else:
            queue.put([30, 10, 0])
    return


def turnToBasket(queue, basket_x):
    if basket_x < 320:
        queue.put([20, 20, 20])
    else:
        queue.put([-20, -20, -20])
    return


def turnRight(queue):
    queue.put([20, 20, 20])
    return


def throw():
    print("should be throwing")
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
blobparams_basket.minArea = 500
blobparams_basket.maxArea = 10000
blobparams_baske=filterByCircularity = False
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
freq = 0.5
speeds_queue = queue.Queue()
mbt = Thread(target=sendMainboard, name="mbt", args=(ser, freq, speeds_queue))

cv2.namedWindow("original")

ball_kp = None
basket_kp = None
ball_img = None
basket_img = None

mbt.start()


while True:
    k = cv2.waitKey(1) & 0xFF
    # press 'q' to exit
    if k == ord('q'):
        break

    frames = getFrames(pipeline)
    if frames:

        if have_ball:
            depth_frame, draw_image, basket_img = getBasketImage(frames)
            basket_kp = getKeypoints(basket_img, basket_detector)
            print("leidsin palli")

            if basket_kp:
                basket_x, basket_y = basket_kp[0].pt[0], basket_kp[0].pt[1]
                basket_dist = depth_frame.get_distance(int(basket_x), int(basket_y))
                print("test")

                if abs(basket_x - 320) < 4:
                    throw()
                    have_ball = False

                else:
                    turnToBasket(speeds_queue, basket_x)

            else:
                turnRight(speeds_queue)

        else:
            draw_image, ball_img = getBallImage(frames)
            ball_kp = getKeypoints(ball_img, ball_detector)


            if ball_kp:
                ball_x, ball_y, ball_size = findNearest(ball_kp)

                #print(ball_y, ball_size)

                if ball_y > 420 and ball_size > 40:
                    have_ball = True

                else:
                    driveToBall(speeds_queue, ball_x)

            else:
                turnRight(speeds_queue)

        draw_image = cv2.drawKeypoints(draw_image, ball_kp, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        draw_image = cv2.drawKeypoints(draw_image, basket_kp, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("original", draw_image)

cv2.destroyAllWindows()
pipeline.stop()
mbt.join()
