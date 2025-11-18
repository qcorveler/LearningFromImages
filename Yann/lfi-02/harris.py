import cv2
import numpy as np
import os

# Load image and convert to gray and floating point
img = cv2.imread('./images/graffiti.png')
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
gray = np.float32(gray)

# Define sobel filter and use cv2.filter2D to filter the grayscale image
Gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
Gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

# Compute G_xx, G_yy, G_xy and sum over all G_xx etc. 3x3 neighbors to compute
Ixx = Gx * Gx
Iyy = Gy * Gy
Ixy,Iyx = Gx * Gy ,Gx * Gy 


kernel = np.ones((3, 3), dtype=np.float32)
Sxx = cv2.filter2D(Ixx, -1, kernel)
Syy = cv2.filter2D(Iyy, -1, kernel)
Sxy = cv2.filter2D(Ixy, -1, kernel)
# entries of the matrix M = \sum_{3x3} [ G_xx Gxy; Gxy Gyy ]

# Note1: this results again in 3 images sumGxx, sumGyy, sumGxy
# Hint: to sum the neighbor values you can again use cv2.filter2D to do this efficiently

# Define parameter
k = 0.04
threshold = 0.01

# Compute the determinat and trace of M using sumGxx, sumGyy, sumGxy. With det(M) and trace(M)
det = (Sxx * Syy) - (Sxy * Sxy)
trace = Sxx + Syy


# you can compute the resulting image containing the harris corner responses with
harris =  det - k * (trace ** 2)

# Filter the harris 'image' with 'harris > threshold*harris.max()'
# this will give you the indices where values are above the threshold.
# These are the corner pixel you want to use

harris_thres = np.zeros_like(harris, dtype=np.uint8)
harris_thres[harris> threshold * harris.max()] = 255

# The OpenCV implementation looks like this - please do not change
harris_cv = cv2.cornerHarris(gray,3,3,k)

# intialize in black - set pixels with corners in with
harris_cv_thres = np.zeros(harris_cv.shape, dtype=np.uint8)
harris_cv_thres[harris_cv > threshold * harris_cv.max()]= 255


# We can add the harris corners on the images 
img_corners = img.copy()
img_corners[harris_thres > 0] = [0, 0, 255]
# just for debugging to create such an image as seen
# in the assignment figure.
# img[harris>threshold*harris.max()]=[255,0,0]


# please leave this - adjust variable name if desired
print("====================================")
print("DIFF:", np.sum(np.absolute(harris_thres - harris_cv_thres)))
print("====================================")

os.makedirs("results/harris", exist_ok=True)
cv2.imwrite("results/harris/Harris_own.png", harris_thres)
cv2.imwrite("results/harris/Harris_cv.png", harris_cv_thres)

cv2.imwrite("results/harris/Image_with_Harris.png", img_corners)
print('Images saved in results/harris/')
cv2.namedWindow('Interactive Systems: Harris Corner')
while True:
    ch = cv2.waitKey(0)
    if ch == 27:
        break

    cv2.imshow('harris',harris_thres)
    cv2.imshow('harris_cv',harris_cv_thres)
cv2.destroyAllWindows()