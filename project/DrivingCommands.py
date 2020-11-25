import time

from Mainboard import mainboardMSG
from ImageThread import ball_x, basket_x, basket_dist

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
