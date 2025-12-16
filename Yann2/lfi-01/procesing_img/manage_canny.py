import cv2 # type: ignore

# Charger l'image et la convertir en niveaux de gris
img_path = r"LearningFromImages\Yann\lfi-01\graffiti.png"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Fenêtre pour l'affichage
cv2.namedWindow('Canny Edge Detection')

# Fonction callback pour le trackbar
def update(x):
    # Récupérer les valeurs des trackbars
    low = cv2.getTrackbarPos('Low Threshold', 'Canny Edge Detection')
    high = cv2.getTrackbarPos('High Threshold', 'Canny Edge Detection')
    
    # Détection des contours
    edges = cv2.Canny(gray, low, high)
    
    # Affichage
    cv2.imshow('Canny Edge Detection', edges)

# Création des trackbars
cv2.createTrackbar('Low Threshold', 'Canny Edge Detection', 50, 500, update)
cv2.createTrackbar('High Threshold', 'Canny Edge Detection', 150, 500, update)

# Appel initial
update(0)
print("Press 'q' to quit.")
# Attente de la touche 'q' pour quitter
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
