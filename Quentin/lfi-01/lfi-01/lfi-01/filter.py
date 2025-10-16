import numpy as np
import cv2


def make_gaussian(size, fwhm = 3, center=None) -> np.ndarray:
    """ Make a square gaussian kernel.

    :param size is the length of a side of the square
    :param fwhm is full-width-half-maximum, which
    :param can be thought of as an effective radius.
    :return np.array
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    k = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return k / np.sum(k)


def convolution_2d(img, kernel) -> np.ndarray:
    """
    Computes the convolution between kernel and image

    :param img: grayscale image
    :param kernel: convolution matrix - 3x3, or 5x5 matrix
    :return: result of the convolution
    """
    # TODO write convolution of arbritrary sized convolution here
    # Attention: convolution should work with kernels of any size.
    # Tip: use Numpy as good as possible, otherwise the algorithm will take too long.
    # Tip: Use Padding, the output image should be the same size of the input without
    #      black pixels at the border  
    # I.e. do not iterate over the kernel, only over the image. The rest goes with Numpy.


    offset = int(kernel.shape[0]/2)
    newimg = np.zeros(img.shape)

    # YOUR CODE HERE

    return newimg


if __name__ == "__main__":

    # 1. load image in grayscale
    

    # image kernels
    sobelmask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelmask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gk = make_gaussian(11)

    # 2 use image kernels

    # 3. compute magnitude of gradients

    # Show resulting images
    # Note: sobel_x etc. must be uint8 images to get displayed correctly astype(np.uint8)
    cv2.imshow("sobel_x", sobel_x)
    cv2.imshow("sobel_y", sobel_y)
    cv2.imshow("mog", mog)
    cv2.waitKey(0)
    cv2.destroyAllWindows()