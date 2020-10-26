import numpy as np
from functools import partial
import cv2

def getValues():
    
    def updateValues(lst, index, value):
        lst[index] = value
        return
    
    choice = input("(0 - ball, 1 - basket1, 2 - basket2): ")
    with open("values.txt", "r") as file:
        lines = file.readlines()
        data = lines[int(choice)].split()
        kernel_size = int(lines[3].split()[1])
        
    lower_limits = np.asarray([int(x) for x in data[1:4]])
    upper_limits = np.asarray([int(x) for x in data[4:]])

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    cv2.namedWindow("original")
    cv2.namedWindow("processed")
    
    cv2.createTrackbar("h_low", "original", int(data[1]), 255, partial(updateValues, lower_limits, 0))
    cv2.createTrackbar("s_low", "original", int(data[2]), 255, partial(updateValues, lower_limits, 1))
    cv2.createTrackbar("v_low", "original", int(data[3]), 255, partial(updateValues, lower_limits, 2))
    cv2.createTrackbar("h_high", "original", int(data[4]), 255, partial(updateValues, upper_limits, 0))
    cv2.createTrackbar("s_high", "original", int(data[5]), 255, partial(updateValues, upper_limits, 1))
    cv2.createTrackbar("v_high", "original", int(data[6]), 255, partial(updateValues, upper_limits, 2))
    
    cap = cv2.VideoCapture(0)
    
    while True:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        thresholded = cv2.inRange(hsv, lower_limits, upper_limits)
        closing = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        
        cv2.imshow("original", frame)
        cv2.imshow("processed", closing)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    with open("values.txt", "w") as file:
        lines[int(choice)] = data[0] + " " + str(lower_limits[0]) + " " + str(lower_limits[1]) + " " + str(lower_limits[2]) + " " + str(upper_limits[0]) + " " + str(upper_limits[1]) + " " + str(upper_limits[2]) + "\n"
        file.writelines(lines)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    getValues()