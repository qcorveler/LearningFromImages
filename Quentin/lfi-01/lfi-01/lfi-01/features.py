import cv2
import numpy as np


def loadImg(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
