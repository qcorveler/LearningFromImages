import cv2 #type: ignore

# Charger l'image et convertir en niveaux de gris
img_path = r"LearningFromImages\Yann\lfi-01\graffiti.png"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Fenêtre pour l'affichage
cv2.namedWindow('Adaptive Thresholding')

# Fonction callback pour le trackbar
def update(x):
    # Récupérer la valeur du trackbar
    ksize = cv2.getTrackbarPos('Block Size', 'Adaptive Thresholding')
    if ksize % 2 == 0:  # Doit être impair
        ksize += 1
    if ksize < 3:
        ksize = 3

    # Adaptive Mean Thresholding
    th_mean = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY,
                                    ksize, 2)
    
    # Adaptive Gaussian Thresholding
    th_gauss = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY,
                                     ksize, 2)
    
    # Affichage côte à côte
    combined = cv2.hconcat([th_mean, th_gauss])
    cv2.imshow('Adaptive Thresholding', combined)

# Trackbar pour changer la taille du voisinage
cv2.createTrackbar('Block Size', 'Adaptive Thresholding', 11, 51, update)

# Initial call
update(0)

cv2.waitKey(10000)
cv2.destroyAllWindows()
