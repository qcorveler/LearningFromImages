import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Impossible d'ouvrir la caméra")

sift = cv2.SIFT_create()
blur_flag = False
rot_flag = False

print("Appuie sur 'b' pour flouter, 'r' pour activer/désactiver la rotation, 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur de capture caméra.")
        break

    # Image originale en couleur
    color = frame.copy()

    # Image en niveaux de gris pour traitement
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        blur_flag = not blur_flag
        print(f"BLUR {'activé' if blur_flag else 'désactivé'}")
    elif key == ord('r'):
        rot_flag = not rot_flag
        print(f"ROTATION {'activée' if rot_flag else 'désactivée'}")

    # Appliquer flou si activé
    if blur_flag:
        gray = cv2.blur(gray, (7, 7))

    # Si rotation activée
    if rot_flag:
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 90, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h))

    # Convertir gray en 3 canaux pour concaténation couleur
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Assembler horizontalement
    combined = np.hstack((color, gray_3ch))

    # Affichage
    cv2.imshow('Gauche: Couleur  |  Droite: Grayscale Rotée', combined)

cap.release()
cv2.destroyAllWindows()
