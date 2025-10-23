# rgb_to_yuv_manual.py
import numpy as np

def rgb_to_yuv_manual(img_rgb):
    """
    Convertit une image RGB en YUV manuellement sans OpenCV.
    Entrée :
        img_rgb : image RGB (valeurs 0–255)
    Sortie :
        yuv_manual : image YUV avec Y, U, V dans [0,255]
    """

    # Normalisation 0-1
    img = img_rgb.astype(np.float32) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Formules de conversion (BT.601)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b

    # Mise à l’échelle pour affichage (0–255)
    yuv = np.zeros_like(img)
    yuv[:, :, 0] = np.clip(y * 255, 0, 255)
    yuv[:, :, 1] = np.clip((u + 0.5) * 255, 0, 255)
    yuv[:, :, 2] = np.clip((v + 0.5) * 255, 0, 255)

    return yuv.astype(np.uint8)


if __name__ == "__main__":
    import cv2 # type: ignore

    img_bgr = cv2.imread("LearningFromImages/Yann/lfi-01/graffiti.png")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    yuv_manual = rgb_to_yuv_manual(img_rgb)
    yuv_cv2 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)

    # Affichage côte à côte
    cv2.imshow("YUV OpenCV", yuv_cv2)
    cv2.imshow("YUV Manual", yuv_manual)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
