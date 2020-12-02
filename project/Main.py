from threading import Thread
import cv2

from DrivingCommands import turnRight, driveToBall, turnToBasket, throw
from ImageThread import imageThread, see_ball, see_basket, basket_x, color_image, ball_morph, basket_morph

have_ball = False

running = True
Thread(name="imageThread", target=imageThread).start()

cv2.namedWindow("original")
cv2.namedWindow("ball_image")
cv2.namedWindow("basket_image")

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
    if (color_image and ball_morph and basket_morph):
        print(color_image)
        cv2.imshow("original", color_image)
        cv2.imshow("ball_image", ball_morph)
        cv2.imshow("basket_image", basket_morph)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        imageThread().join()
        running = False


