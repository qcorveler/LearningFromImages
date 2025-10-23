import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

def fourierTransform(img, offset=50,want_save:bool = False,want_plot:bool=True,want_print:bool=True):

    f = np.fft.fft2(img)
    # Print of the content of a pixel
    if want_print:
        print("FFT of the image at (100,150):")
        print("Complex value:", complex(round(np.real(f[100,150]), 2), round(np.imag(f[100,150]), 2)))
        print("Real part:", round(np.real(f[100,150]), 2))
        print("Imaginary part:", round(np.imag(f[100,150]), 2))
        print("Magnitude :", round(np.abs(f[100,150]), 2))
        print("Angle: ", round(np.angle(f[100,150]), 2), "rad")
        
        # Note that the zero-frequency component will be located at top-left corner
        print("\n zero-frequency component : ",  complex(round(np.real(f[0,0]), 2), round(np.imag(f[0,0]), 2)))

    # Now, we center the zero-frequency compenent
    fshift = np.fft.fftshift(f)
    # Make sur that he's at the right place
    if want_print:
        print("Centered zero-frequency component : ",  complex(round(np.real(fshift[img.shape[0]//2, img.shape[1]//2]), 2), round(np.imag(fshift[img.shape[0]//2, img.shape[1]//2]), 2)))

    # Now we calculate the magnitude spectrum (use abs value and then 20*log)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) )

    # Create of a filter mask (low-pass filter)
    rows, cols   = img.shape
    crow, ccol = rows//2, cols//2
    offset = offset
    mask = np.zeros((rows,cols), np.uint8)
    mask[crow-offset:crow+offset, ccol-offset:ccol+offset] = 1
    # Apply mask and inverse DFT
    fshift_filtered = fshift*mask   # We keep only the low freqencies in the mask area (near the middle)
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_filt = np.fft.ifft2(f_ishift)
    img_filt = np.abs(img_filt)

    if want_save:
        img_norm = cv.normalize(img_filt, None, 0, 255, cv.NORM_MINMAX)
        img_uint8 = np.uint8(img_norm)
        cv.imwrite("LearningFromImages\Yann\lfi-01\Filtered_image.png", img_uint8)    



    if want_plot:
        plt.figure(figsize=(12,4))
        plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Original Image'), plt.axis('off')
        plt.subplot(222), plt.imshow(20*np.log(1+np.abs(f)), cmap='gray'), plt.title('Fourier tranformation'), plt.axis('off')
        plt.subplot(223), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Magnitude Spectrum'), plt.axis('off')
        plt.subplot(224), plt.imshow(img_filt, cmap='gray'), plt.title('Filtered Image'), plt.axis('off')
        plt.show()


if __name__ == "__main__":
    imgPath = "LearningFromImages/Yann/lfi-01/graffiti.png"
    img= cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    
    fourierTransform(img,
                     want_plot=True,
                     want_save=True,
                     want_print=False)
    