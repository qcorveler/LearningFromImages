import cv2 #type: ignore

# Question a
img = cv2.imread('LearningFromImages/Yann/lfi-01/graffiti.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

cv2.imshow('HSV', hsv)
cv2.imshow('LAB', lab)
cv2.imshow('YUV', yuv)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# Question b
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Adaptive Gaussian Thresholding
gaussian = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2)

# Otsu Thresholding
_, otsu = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('Gaussian', gaussian)
cv2.imshow('Otsu', otsu)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# Question c
edges = cv2.Canny(gray, 100, 200)
edges = cv2.Canny(img, 100, 200)

cv2.imshow('Canny Edges (gray)', edges)
cv2.imshow('Canny Edges (RGB)', edges)

cv2.waitKey(10000)
cv2.destroyAllWindows()
