# rgb_to_lab_manual.py
import numpy as np

def rgb_to_lab_manual(img_rgb):
    """
    Convertit une image RGB en CIE LAB manuellement sans OpenCV.
    Entrée :
        img_rgb : image RGB (valeurs 0–255)
    Sortie :
        lab_manual : image LAB avec L [0–100], a,b ≈ [-128,127]
    """

    # --- Étape 1 : normalisation et passage en linéaire ---
    img = img_rgb.astype(np.float32) / 255.0
    mask = img > 0.04045
    img_lin = np.zeros_like(img)
    img_lin[mask] = ((img[mask] + 0.055) / 1.055) ** 2.4
    img_lin[~mask] = img[~mask] / 12.92

    # --- Étape 2 : conversion RGB -> XYZ (D65) ---
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = np.dot(img_lin, M.T)

    # --- Étape 3 : normalisation par référence blanche D65 ---
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    X, Y, Z = xyz[:, :, 0] / Xn, xyz[:, :, 1] / Yn, xyz[:, :, 2] / Zn

    # --- Étape 4 : fonction f(t) ---
    epsilon, kappa = 0.008856, 903.3
    fX = np.where(X > epsilon, np.cbrt(X), (kappa * X + 16) / 116)
    fY = np.where(Y > epsilon, np.cbrt(Y), (kappa * Y + 16) / 116)
    fZ = np.where(Z > epsilon, np.cbrt(Z), (kappa * Z + 16) / 116)

    # --- Étape 5 : calcul de L, a, b ---
    L = (116 * fY) - 16
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)

    lab = np.dstack((L, a, b)).astype(np.float32)
    return lab


if __name__ == "__main__":
    import cv2 # type: ignore

    img_bgr = cv2.imread("LearningFromImages/Yann/lfi-01/graffiti.png")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    lab_manual = rgb_to_lab_manual(img_rgb)
    lab_cv2 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # Mise à l’échelle pour affichage (optionnelle)
    lab_manual_display = np.clip((lab_manual - lab_manual.min()) / (lab_manual.max() - lab_manual.min()) * 255, 0, 255).astype(np.uint8)

    cv2.imshow("LAB OpenCV", lab_cv2)
    cv2.imshow("LAB Manual", lab_manual_display)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
