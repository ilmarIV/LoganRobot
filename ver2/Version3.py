#import json
import cv2
import pyrealsense2 as rs
import numpy as np
import time
from serial import Serial
from threading import Thread
import queue
#import websocket

"""
def clientFunction(stop_queue, game_queue, name, ws):
    while True:
        received = ws.recv()

        if received != "":
            info = json.loads(received)

            if name in info["targets"]:

                if info["signal"] == "stop":
                    stop_queue.put(True)

                elif info["signal"] == "start":
                    target = info["targets"].index(name) + 1
                    game_queue.put(target)
"""


stop_queue = queue.Queue()
#game_queue = queue.Queue()

#host = "ws://192.168.43.105"
#name = "Logan"
#ws = websocket.create_connection(host)

#sckt = Thread(target=clientFunction, name="sckt", args=(stop_queue, game_queue, name, ws))
#sckt.start()

choice = int(input("(1 - pink basket, 2 - blue basket): "))
"""
print("waiting for commands...")
while game_queue.empty():
    continue

choice = game_queue.get()
print(choice)
"""
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
        if not stop_queue.empty():
            return

        if not speeds_queue.empty():
            speeds = speeds_queue.get()

            byte_arr = bytearray([*speeds, *[170]])

            ser.write(byte_arr)
            time.sleep(0.1)

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
    # a = [190]
    if ball_y < 420:
        for y in range(int(ball_y) + 50, 479, 1):
            # a.append(frame[y][int(ball_x)])

            if frame[y][int(ball_x)] < 105 and frame[y - 1][int(ball_x)] < 105:
                # print("false")
                return False
    # print(max(a), min(a))
    return True


def reverse(queue, speed):
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

    queue.put([180, 180, 180, 0])

    return


def throw(speeds_queue, basket_dist, basket_x):
    throw_speed = mapThrowSpeed(basket_dist)

    if basket_x < 422:
        speeds_queue.put([190, 125, 50, throw_speed])

    elif basket_x > 422:
        speeds_queue.put([190, 125, 70, throw_speed])

    else:
        speeds_queue.put([170, 125, 50, throw_speed])

    return


def mapThrowSpeed(basket_dist):
    """
    The solution we had for mapping basket distances and thrower speeds, we measured some distances and made linear functions inbetween them.
    """
    
    """
    if basket_dist < 0.49:
        speed = 316.33 * basket_dist + 1010

    elif basket_dist >= 0.49 and basket_dist < 0.60:
        speed = 118.18 * basket_dist + 1130.09

    elif basket_dist >= 0.60 and basket_dist < 0.70:
        speed = 92 * basket_dist + 1116

    elif basket_dist >= 0.70 and basket_dist < 0.79:
        speed = 146.44 * basket_dist + 1077.89

    elif basket_dist >= 0.79 and basket_dist < 0.93:
        speed = 55 * basket_dist + 1144.86

    elif basket_dist >= 0.93 and basket_dist < 1.00:
        speed = 100 * basket_dist + 1105

    elif basket_dist >= 1.00 and basket_dist < 1.16:
        speed = 91 * basket_dist + 1183

    elif basket_dist >= 1.16 and basket_dist < 1.28:
        speed = 155 * basket_dist + 1021

    elif basket_dist >= 1.28 and basket_dist < 1.38:
        speed = 102.5 * basket_dist + 1105.2

    elif basket_dist >= 1.38 and basket_dist < 1.51:
        speed = 78 * basket_dist + 1143.85

    elif basket_dist >= 1.51 and basket_dist < 1.67:
        speed = 55 * basket_dist + 1170.06

    elif basket_dist >= 1.67 and basket_dist < 1.83:
        speed = 62.5 * basket_dist + 1165.63

    elif basket_dist >= 1.83 and basket_dist < 2.02:
        speed = 78 * basket_dist + 1139

    elif basket_dist >= 2.02 and basket_dist < 2.13:
        speed = -105872.73 * basket_dist + 226813.91

    elif basket_dist >= 2.13 and basket_dist < 2.36:
        speed = 90.65 * basket_dist + 1997.26

    elif basket_dist >= 2.36 and basket_dist < 3.07:
        speed = 25 * basket_dist + 1308.61

    elif basket_dist >= 3.07 and basket_dist < 3.27:
        speed = 172.5 * basket_dist + 834.1

    else:
        speed = 1400
    """
    #The solution we had to use for competition, does not actually work properly
    NewValue = 15 * basket_dist ** 2 + 40

    return int(NewValue)


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
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)

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
throwing = False
throw_start = False

cv2.namedWindow("original")
cv2.namedWindow("ball")
cv2.namedWindow("basket")
cv2.namedWindow("gray")

ball_kp = None
basket_kp = None
ball_img = None
basket_img = None

thrower_stop = 0

# reverse speed
reverse_speed = [55, 125, 195, thrower_stop]

# slow driving speeds
forward_slow = [190, 125, 60, thrower_stop]  # tehtud
right_slow = [200, 125, 80, thrower_stop]  # tehtud
left_slow = [170, 125, 50, thrower_stop]  # tehtud

# fast driving speeds
forward_fast = [250, 125, 0, thrower_stop]  # tehtud
right_fast = [250, 125, 60, thrower_stop]  # tehtud
left_fast = [190, 125, 0, thrower_stop]  # tehtud

# turn right speed
right_turn = [150, 150, 150, thrower_stop]

# turn around ball
around_w_basket_right_1_slow = [125, 137, 125, thrower_stop]
around_w_basket_left_1_slow = [123, 137, 122, thrower_stop]

around_w_basket_right_2_slow = [128, 113, 127, thrower_stop]
around_w_basket_left_2_slow = [125, 113, 125, thrower_stop]

around_w_basket_right_1 = [125, 150, 125, thrower_stop]
around_w_basket_left_1 = [117, 150, 97, thrower_stop]

around_w_basket_right_2 = [148, 100, 128, thrower_stop]
around_w_basket_left_2 = [125, 100, 125, thrower_stop]

around_wo_basket_right = [125, 165, 125, thrower_stop]
around_wo_basket_left = [120, 165, 105, thrower_stop]

# around_wo_basket_right = [125, 140, 125, thrower_stop]
# around_wo_basket_left = [123, 140, 110, thrower_stop]


mbt.start()

#Test robots thrower needed to be configured this way, proble fixed with new electronics
"""
not_moving = [125, 125, 125, 0]
yes_moving = [125, 125, 125, 0]

for i in range(2):
    ser.write(bytearray(yes_moving))
    time.sleep(1)
    ser.write(bytearray(not_moving))
    time.sleep(1)
"""

while True:

    if (cv2.waitKey(1) & 0xFF) == ord('q') or not stop_queue.empty():
        break

    frames = getFrames(pipeline)

    if frames:

        draw_image, depth_frame, ball_img, basket_img, gray_image = getImages(frames)

        basket_kp = getKeypoints(basket_img, basket_detector)
        if basket_kp:
            basket_x, basket_y = basket_kp[0].pt[0], basket_kp[0].pt[1]

            distances = []
            for i in range(0, 480, 8):
                for j in range(0, 848, 8):
                    if basket_img[i][j] == 255:
                        distances.append(depth_frame.get_distance(j, i))
            basket_dist = sum(distances) / len(distances)

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

        if throwing:
            if throw_start:
                while not speeds_queue.empty():
                    speeds_queue.get()

                print(basket_dist)
                start = time.time()
                throw_start = False

            if time.time() - start <= 1.8 and basket_x:
                throw(speeds_queue, basket_dist, basket_x)

            else:
                throwing = False

        elif have_ball:

            if (ball_kp and ball_x > 0) and ball_y > 460:
                reverse(speeds_queue, reverse_speed)

            elif (ball_kp and ball_x > 0) and ball_y > 370:

                if basket_kp:
                    if abs(basket_x - 424) < 10 and abs(ball_x - 424) < 4:
                        throw_start = True
                        throwing = True

                    elif abs(basket_x - 424) < 30 and abs(ball_x - 424) < 10:
                        rotateToBasket(speeds_queue, ball_x, basket_x, around_w_basket_right_1_slow,
                                       around_w_basket_left_1_slow, around_w_basket_right_2_slow,
                                       around_w_basket_left_2_slow)

                    else:
                        rotateToBasket(speeds_queue, ball_x, basket_x, around_w_basket_right_1, around_w_basket_left_1,
                                       around_w_basket_right_2, around_w_basket_left_2)

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

        draw_image = cv2.drawKeypoints(draw_image, ball_kp, np.array([]), (0, 0, 255),
                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        draw_image = cv2.drawKeypoints(draw_image, basket_kp, np.array([]), (0, 0, 255),
                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow("original", draw_image)
        cv2.imshow("ball", ball_img)
        cv2.imshow("basket", basket_img)
        cv2.imshow("gray", gray_image)

cv2.destroyAllWindows()
pipeline.stop()
stop_queue.put(True)
ser.write(bytearray([125, 125, 125, 0, 170]))
mbt.join()
#sckt.join()
