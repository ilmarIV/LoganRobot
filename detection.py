import numpy as np
import cv2

def loop():
    with open("project/values.txt", "r") as file:
        lines = file.readlines()
        
        ball_lower = np.asarray([int(x) for x in lines[0].split()[1:4]])
        ball_upper = np.asarray([int(x) for x in lines[0].split()[4:]])
        
        basket1_lower = np.asarray([int(x) for x in lines[1].split()[1:4]])
        basket1_upper = np.asarray([int(x) for x in lines[1].split()[4:]])
        
        basket2_lower = np.asarray([int(x) for x in lines[2].split()[1:4]])
        basket2_upper = np.asarray([int(x) for x in lines[2].split()[4:]])
        
        kernel_size = int(lines[3].split()[1])
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    blobparams_ball = cv2.SimpleBlobDetector_Params()
    blobparams_ball.filterByArea = True
    blobparams_ball.minArea = 10
    blobparams_ball.maxArea = 10000
    blobparams_ball.filterByCircularity = True
    blobparams_ball.filterByConvexity = False
    blobparams_ball.filterByColor = False
    blobparams_ball.filterByInertia = False
    
    blobparams_basket = cv2.SimpleBlobDetector_Params()
    blobparams_basket.filterByArea = True
    blobparams_basket.minArea = 10
    blobparams_basket.maxArea = 10000
    blobparams_basket.filterByCircularity = False
    blobparams_basket.filterByConvexity = False
    blobparams_basket.filterByColor = False
    blobparams_basket.filterByInertia = False
    
    ball_detector = cv2.SimpleBlobDetector_create(blobparams_ball)
    basket1_detector = cv2.SimpleBlobDetector_create(blobparams_basket)
    basket2_detector = cv2.SimpleBlobDetector_create(blobparams_basket)
    
    cv2.namedWindow("original")
    cv2.namedWindow("ball_image")
    cv2.namedWindow("basket1_image")
    cv2.namedWindow("basket2_image")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        _, frame = cap.read()
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        ball_thresholded = cv2.inRange(hsv_image, ball_lower, ball_upper)
        ball_morph = cv2.morphologyEx(ball_thresholded, cv2.MORPH_CLOSE, kernel)
        
        basket1_thresholded = cv2.inRange(hsv_image, basket1_lower, basket1_upper)
        basket1_morph = cv2.morphologyEx(basket1_thresholded, cv2.MORPH_CLOSE, kernel)
        
        basket2_thresholded = cv2.inRange(hsv_image, basket2_lower, basket2_upper)
        basket2_morph = cv2.morphologyEx(basket2_thresholded, cv2.MORPH_CLOSE, kernel)
        
        ball_keypoints = ball_detector.detect(ball_morph)
        basket1_keypoints = basket1_detector.detect(basket1_morph)
        basket2_keypoints = basket2_detector.detect(basket2_morph)
        
        frame = cv2.drawKeypoints(frame, ball_keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        frame = cv2.drawKeypoints(frame, basket1_keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        frame = cv2.drawKeypoints(frame, basket2_keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        if(ball_keypoints):
            for ball_keypoint in ball_keypoints:
                ball_x = ball_keypoint.pt[0]
                ball_y = ball_keypoint.pt[1]
                ball_size = ball_keypoint.size


        cv2.imshow("original", frame)
        cv2.imshow("ball_image", ball_morph)
        cv2.imshow("basket1_image", basket1_morph)
        cv2.imshow("basket2_image", basket2_morph)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    loop()