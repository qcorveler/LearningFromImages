import cv2 as cv
import numpy as np
from fourier import fourierTransform


cap = cv.VideoCapture(0)
cv.namedWindow('Learning from images: SIFT feature visualization')
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur: impossible de lire la caméra")
        break
    # 1. read each frame from the camera (if necessary resize the image)
    #    and extract the SIFT features using OpenCV methods
    #    Note: use the gray image - so you need to convert the image
    gray = cv.cvtColor(cap.read()[1], cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp = sift.detect(gray, None)
    # 2. draw the keypoints using cv2.drawKeypoints
    #    There are several flags for visualization - e.g. DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    img=cv.drawKeypoints(
        gray,
        kp,
        None,
        color= (0,255,0),
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('SIFT feature vizualisation', img)

    # close the window and application by pressing a key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()