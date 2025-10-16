import numpy as np
import cv2

def cameraCapture():
    cap = cv2.VideoCapture(0)
    mode = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # wait for key and switch to mode
        ch = cv2.waitKey(1) & 0xFF
        if ch == ord('0'):
            mode = 0
        if ch == ord('1'):
            mode = 1
        if ch == ord('2'):
            mode = 2
        if ch == ord('3'):
            mode = 3
        if ch == ord('4'):
            mode = 4
        # ...

        if ch == ord('q'):
            break

        if mode == 1:
            # just example code
            # your code should implement
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
        if mode == 2:
            frame = cv2.GaussianBlur(frame, (11, 11), 0)
        if mode == 3:
            frame = cv2.GaussianBlur(frame, (15, 15), 0)
        if mode == 4:
            frame = cv2.GaussianBlur(frame, (21, 21), 0)

        # Display the resulting frame
        cv2.imshow('frame', frame)


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def loadImg(path : str) :
    image = cv2.imread(path)
    print(image)
    image_gray = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)
    print(image_gray)
    newImage = np.concatenate((image, image_gray), axis=1)
    cv2.imshow("IMAGE", newImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__" :
    loadImg("graffiti.png")