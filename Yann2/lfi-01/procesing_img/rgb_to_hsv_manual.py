# rgb_to_hsv_manual.py
import numpy as np
import cv2 # type: ignore

def rgb_to_hsv_manual(img_rgb):
    """
    Convertit une image RGB en HSV manuellement sans OpenCV.
    Entrée :
        img_rgb : image RGB (valeurs 0-255)
    Sortie :
        hsv_manual : image HSV avec H[0-179], S[0-255], V[0-255] (comme OpenCV)
    """

    # Normalisation entre 0 et 1
    rgb = img_rgb.astype(np.float32) / 255.0
    b , g ,r = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Calcul des composantes principales
    cmax = np.max(rgb, axis=2)
    cmin = np.min(rgb, axis=2)
    diff = cmax - cmin

    # --- HUE ---
    h = np.zeros_like(cmax)

    mask = (diff != 0)
    rmask = (cmax == r) & mask
    gmask = (cmax == g) & mask
    bmask = (cmax == b) & mask

    h[rmask] = (60 * ((g[rmask] - b[rmask]) / diff[rmask]) + 360) % 360
    h[gmask] = (60 * ((b[gmask] - r[gmask]) / diff[gmask]) + 120) % 360
    h[bmask] = (60 * ((r[bmask] - g[bmask]) / diff[bmask]) + 240) % 360

    # Conversion H en [0,179] pour OpenCV
    h = (h / 2).astype(np.uint8)

    # --- SATURATION ---
    s = np.zeros_like(cmax)
    s[cmax != 0] = (diff[cmax != 0] / cmax[cmax != 0])
    s = (s * 255).astype(np.uint8)

    # --- VALUE ---
    v = (cmax * 255).astype(np.uint8)

    hsv_manual = cv2.merge([h, s, v])
    return hsv_manual


if __name__ == "__main__":
    img_rgb = cv2.imread('LearningFromImages/Yann/lfi-01/graffiti.png')
    if img_rgb is None:
        print("Erreur : impossible de charger l'image.")
        exit()

    hsv_manual = rgb_to_hsv_manual(img_rgb)
    print("Hue max value (manual):", np.max(hsv_manual[:, :, 0]))
    hsv_cv2 = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    print("Hue max value (openCV):", np.max(hsv_cv2[:, :, 0]))

    print("Hue difference between manual and openCV:", np.abs(hsv_manual[:,:,0].astype(float) - hsv_cv2[:,:,0].astype(float)).mean())

    cv2.imshow("Original Image", img_rgb)
    cv2.imshow("HSV openCV", hsv_cv2)
    cv2.imshow("HSV manual", hsv_manual)

    # # Affichage côte à côte
    # combined = np.hstack((
    #     cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
    #     cv2.cvtColor(hsv_cv2, cv2.COLOR_HSV2RGB),
    #     cv2.cvtColor(hsv_manual, cv2.COLOR_HSV2RGB)
    # ))

    # cv2.imshow("Left=Original | Middle=OpenCV HSV->RGB | Right=Manual HSV->RGB", combined)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
