import time
from serial import Serial

from variables import *

turn_speed = 20
speed = 40

Ku = 1
Tu = 1

Kp = 0.6 * Ku
Ki = (1.2 * Ku) / Tu
Kd = (3 * Ku * Tu) / 40

last_error = 0
integral = 0
seconds = time.time()

#basket_x = None
#ball_x = None


ser = Serial('/dev/ttyACM0', baudrate=115200, timeout=1)

def mainboardMSG(speed_1, speed_2, speed_3, ser):
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


def turnRight(ser):
    mainboardMSG(turn_speed, 0, -turn_speed, ser)


def turnToBasket(basket_x, ser):
    if basket_x < 320:
        mainboardMSG(turn_speed, 0, -turn_speed, ser)
    else:
        mainboardMSG(-turn_speed, 0, turn_speed, ser)


def throw():
    pass