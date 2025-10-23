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

def loadImg(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def concatenateImgs(image1, image2):
    return np.concatenate((image1, image2), axis=1) # concatenate the 2 images

def concatenateImgColorAndGray(image, gray_image) :
    image_gray = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB) # Change the representation matrix from 2D to 3D
    newImage = concatenateImgs(image, image_gray)
    loadImg("Concatenate images", newImage)

def changeColorSpace(image, color_conversion_code):
    newImage = cv2.cvtColor(image, color_conversion_code)
    loadImg(f"transformed image with conversion {color_conversion_code}", newImage)
    return newImage

def thresholding(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding	
    _, thresh_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    loadImg("binary threshold", thresh_img)    

    # Apply Adaptive Mean Thresholding
    thresh_img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    loadImg("adaptative Mean thresholding", thresh_img)

    # Apply Adaptive Gaussian Thresholding 
    thresh_img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    loadImg("adaptative Gaussian thresholding", thresh_img)

    # Apply Adaptive Otsu Thresholding 
    _, thresh_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    loadImg("adaptative Otsu thresholding", thresh_img)

def sobel_filter(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) # horizontal edges
    sobely = sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges
    both = concatenateImgs(sobelx, sobely)
    loadImg("horizontal edges (left) and vertical edges (right)", both)
    
    # Compute gradient magnitude
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    
    # Convert to uint8
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    loadImg("Edges detection using sobel", gradient_magnitude)

def canny(image):
    # Apply Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(image, (5, 5), 1.4)
 
    # Apply Canny Edge Detector
    edges = cv2.Canny(blur, threshold1=100, threshold2=200)

    loadImg("Edges Detection using canny", edges)

if __name__ == "__main__" :
    image = cv2.imread("graffiti.png") # load the image without modification
    gray_image = cv2.imread("graffiti.png", cv2.IMREAD_GRAYSCALE) # load the image grayscaled
    concatenateImgColorAndGray(image, gray_image)
    changeColorSpace(image, cv2.COLOR_BGR2RGB)
    changeColorSpace(image, cv2.COLOR_BGR2HSV)
    print("LAB")
    newImage = changeColorSpace(image, cv2.COLOR_BGR2LAB)
    print("BGR")
    changeColorSpace(newImage, cv2.COLOR_LAB2BGR)
    changeColorSpace(image, cv2.COLOR_BGR2YUV)
    thresholding(image)
    sobel_filter(image)
    canny(image)