import numpy as np
import cv2
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

###############################################################
#
# Write your own descriptor / Histogram of Oriented Gradients
#
###############################################################


def plot_histogram(hist, bins):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()


def compute_simple_hog(imgcolor, keypoints, mag_eps=1e-3):

    # convert color to gray image and extract feature in gray
    img_gray = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2GRAY)

    # compute x and y gradients (sobel kernel size 5)
    grad_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=5)

    # compute magnitude and angle of the gradients
    magnitude = cv2.magnitude(grad_x, grad_y)
    angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)

    # go through all keypoints and and compute feature vector
    descr = np.zeros((len(keypoints), 8), np.float32)
    count = 0
    for kp in keypoints:
        # print kp.pt, kp.size
        print(kp.pt, kp.size)
        # extract angle in keypoint sub window
        cx, cy = int(kp.pt[0]), int(kp.pt[1])
        half = int(kp.size // 2)
        angle_patch = angle[cy - half:cy + half + 1, cx - half:cx + half + 1]


        # extract gradient magnitude in keypoint subwindow
        magnitude_patch = magnitude[cy - half:cy + half + 1, cx - half:cx + half + 1]

        # create histogram of angle in subwindow BUT only where magnitude of gradients is non zero! Why? Find an
        # answer to that question use np.histogram
        mask = magnitude_patch > mag_eps
        ang_valid = angle_patch[mask]
        mag_valid = magnitude_patch[mask]
        hist, bins = np.histogram(ang_valid, bins=8, range=(0, 360), weights=mag_valid)
        hist /= (np.linalg.norm(hist) + 1e-6)

        plot_histogram(hist, bins)

        descr[count] = hist
        count+=1

    return descr


keypoints = [cv2.KeyPoint(15, 15, 11)]

# test for all test images
test = cv2.imread('./images/hog_test/horiz.jpg')
descriptor = compute_simple_hog(test, keypoints)

