import cv2
import numpy as np


def loadImg(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

### Convolution function
def convolve2d(image, kernel):
    """Convolution 2D d'une image grayscale avec un noyau, sans fonctions externes."""
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    # Padding symétrique pour conserver la taille
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    # Préparer une image de sortie vide
    output = np.zeros_like(image, dtype=float)

    # Convolution manuelle
    for i in range(img_h):
        for j in range(img_w):
            region = padded[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum(region * kernel)
    return output


# APPLICATION 
### SIFT
cap = cv2.VideoCapture(0)

sift = cv2.SIFT_create()

while True:

    # 1. read each frame from the camera (if necessary resize the image)
    #    and extract the SIFT features using OpenCV methods
    #    Note: use the gray image - so you need to convert the image
    # 2. draw the keypoints using cv2.drawKeypoints
    #    There are several flags for visualization - e.g. DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

    # close the window and application by pressing a key


    _, frame = cap.read()

    # Conversion en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Détection des keypoints et calcul des descripteurs ---
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # --- Dessiner les keypoints sur l'image ---
    frame_with_keypoints = cv2.drawKeypoints(
        frame, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    # --- Afficher le résultat ---
    cv2.imshow('Learning from images: SIFT feature visualization', frame_with_keypoints)

    # Quitter avec n'importe quelle touche (255 veut dire 'aucune touche')
    if cv2.waitKey(1) & 0xFF != 255 :
        break

cap.release()
cv2.destroyAllWindows()

### CONVOLUTION
gaussian_kernel = (1/273) * np.array([
    [1,  4,  7,  4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1,  4,  7,  4, 1]
])

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=float)

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=float)

image = cv2.imread('graffiti.png', cv2.IMREAD_GRAYSCALE)
loadImg("Image normale", image)
blurred = convolve2d(image, gaussian_kernel)

blurred = blurred.astype(np.uint8)
loadImg("blurred", blurred)

grad_x = convolve2d(blurred, sobel_x)
grad_y = convolve2d(blurred, sobel_y)

# Magnitude du gradient
gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Normalisation (0–255 pour affichage)
gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude)) * 255
gradient_magnitude = gradient_magnitude.astype(np.uint8)

loadImg('Gradient Magnitude (after sobelx and y convolution)', gradient_magnitude)