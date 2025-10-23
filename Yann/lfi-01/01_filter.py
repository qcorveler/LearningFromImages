import numpy as np
import cv2 # type: ignore


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
    # TO DO write convolution of arbritrary sized convolution here
    # Attention: convolution should work with kernels of any size.
    # Tip: use Numpy as good as possible, otherwise the algorithm will take too long.
    # Tip: Use Padding, the output image should be the same size of the input without
    #      black pixels at the border  
    # I.e. do not iterate over the kernel, only over the image. The rest goes with Numpy.


    offset = int(kernel.shape[0]/2)
    print(offset)
    newimg = np.zeros(img.shape)

    padded_img = np.pad(img, pad_width=offset, mode='edge')
    newimg = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region_padded_img = padded_img[i:i+2*offset+1, j:j+2*offset+1]
            newimg[i,j] = np.sum(region_padded_img * kernel)
        
    return newimg


if __name__ == "__main__":
    
    # 1. load image in grayscale
    img = cv2.imread("Yann/lfi-01/graffiti.png", cv2.IMREAD_GRAYSCALE)

    # image kernels
    sobelmask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelmask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gk = make_gaussian(51)
    # Apply Gaussian filter
    img = convolution_2d(img, gk)
    # 2 use image kernels
    sobel_x = convolution_2d(img, sobelmask_x)
    sobel_y = convolution_2d(img, sobelmask_y)
    # 3. compute magnitude of gradients
    mog = np.sqrt(sobel_x**2 + sobel_y**2)
    # Show resulting images
    # Note: sobel_x etc. must be uint8 images to get displayed correctly astype(np.uint8)
    cv2.imshow("Original image", img)
    cv2.imshow("sobel_x", sobel_x.astype(np.uint8)) 
    cv2.imshow("sobel_y", sobel_y.astype(np.uint8)) 
    cv2.imshow("mog", mog.astype(np.uint8))         
    cv2.waitKey(0)
    cv2.destroyAllWindows()