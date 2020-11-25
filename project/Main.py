from threading import Thread
import cv2

from DrivingCommands import turnRight, driveToBall, turnToBasket, throw
from ImageThread import imageThread, see_ball, see_basket, basket_x

have_ball = False

running = True
Thread(name="imageThread", target=imageThread).start()

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
        running = False


