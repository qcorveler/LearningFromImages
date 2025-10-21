import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def loadImg(name, image):
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


img = cv.imread('graffiti.png', cv.IMREAD_GRAYSCALE)

loadImg("Noir et blanc", img)

ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)

magnitude_spectrum = 20*np.log(np.abs(ft_shift))

plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Image')
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Magnitude Spectrum')
plt.show()

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2


# filter on low frequencies
# créer un masque circulaire centré
mask = np.zeros((rows, cols), np.uint8)
r = 100  # rayon du filtre
cv.circle(mask, (ccol, crow), r, 1, thickness=-1)

# appliquer le masque
fshift_filtered = ft_shift * mask
magnitude_spectrum_filtered = 20*np.log(np.abs(fshift_filtered))
plt.subplot(111), plt.imshow(magnitude_spectrum_filtered, cmap='gray'), plt.title('Magnitude Spectrum filtered')
plt.show()

# Inverse shift
f_ishift = np.fft.ifftshift(fshift_filtered)

# Transformée inverse de Fourier
img_filtered = np.fft.ifft2(f_ishift)
img_filtered = np.abs(img_filtered)
cv.imwrite('Filtre passe bas.png', img_filtered)


# filter on high frequencies
mask = np.ones((rows, cols), np.uint8)
cv.circle(mask, (ccol, crow), 30, 0, thickness=-1)
fshift_filtered = ft_shift * mask

# Inverse shift
f_ishift = np.fft.ifftshift(fshift_filtered)

# Transformée inverse de Fourier
img_filtered = np.fft.ifft2(f_ishift)
img_filtered = np.abs(img_filtered)
cv.imwrite('filtre passe haut.png', img_filtered)