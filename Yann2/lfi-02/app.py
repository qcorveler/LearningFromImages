import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
st.set_page_config(page_title="Learning From Images - LFI-02", layout="wide")

# Title and description
st.title("🖼️ Learning From Images - LFI-02")
st.markdown("""
Exploration des techniques de traitement d'images :
- **Harris Corner Detection** : Détection automatique de coins dans les images
- **Simple HOG** : Histogramme des gradients orientés
- **Image Retrieval** : Récupération d'images basée sur les descripteurs
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Harris Corners", "Simple HOG", "Image Retrieval"])

# ================== TAB 1: HARRIS CORNERS ==================
with tab1:
    st.header("Harris Corner Detection")
    st.markdown("""
    La détection de coins Harris identifie les pixels de l'image où il y a une forte variation 
    d'intensité dans deux directions perpendiculaires (les coins).
    
    Algorithme:
    1. Calculer les gradients Gx et Gy avec l'opérateur Sobel
    2. Calculer la matrice M = [Ixx Ixy; Ixy Iyy]
    3. Appliquer la réponse Harris: R = det(M) - k * trace(M)²
    4. Seuillage pour obtenir les coins
    """)
    
    # Load and process image
    img_path = "images/graffiti.png"
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_float = np.float32(gray)
        
        # Compute gradients
        Gx = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
        Gy = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute Harris response
        Ixx = Gx * Gx
        Iyy = Gy * Gy
        Ixy = Gx * Gy
        
        kernel = np.ones((3, 3), dtype=np.float32)
        Sxx = cv2.filter2D(Ixx, -1, kernel)
        Syy = cv2.filter2D(Iyy, -1, kernel)
        Sxy = cv2.filter2D(Ixy, -1, kernel)
        
        k = 0.04
        det = (Sxx * Syy) - (Sxy * Sxy)
        trace = Sxx + Syy
        harris = det - k * (trace ** 2)
        
        # OpenCV implementation
        harris_cv = cv2.cornerHarris(gray_float, 3, 3, k)
        
        # Threshold slider
        threshold = st.slider("Seuil de détection", 0.0, 1.0, 0.01, step=0.01)
        
        # Thresholding
        harris_thres = np.zeros_like(harris, dtype=np.uint8)
        harris_thres[harris > threshold * harris.max()] = 255
        
        harris_cv_thres = np.zeros_like(harris_cv, dtype=np.uint8)
        harris_cv_thres[harris_cv > threshold * harris_cv.max()] = 255
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image originale")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width =True)
        
        with col2:
            st.subheader("Harris Response Heatmap")
            fig, ax = plt.subplots()
            im = ax.imshow(harris, cmap='hot')
            plt.colorbar(im, ax=ax)
            ax.set_title("Réponse Harris")
            st.pyplot(fig)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Implémentation manuelle")
            img_corners = img.copy()
            corners = np.argwhere(harris_thres > 0)
            for corner in corners:
                y, x = corner
                cv2.circle(img_corners, (x, y), 5, (0, 255, 0), -1)
            st.image(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB), use_container_width =True)
        
        with col4:
            st.subheader("OpenCV Implementation")
            img_corners_cv = img.copy()
            corners_cv = np.argwhere(harris_cv_thres > 0)
            for corner in corners_cv:
                y, x = corner
                cv2.circle(img_corners_cv, (x, y), 5, (0, 255, 0), -1)
            st.image(cv2.cvtColor(img_corners_cv, cv2.COLOR_BGR2RGB), use_container_width =True)
        
        st.info(f"Coins détectés (manuel): {len(corners)} | Coins détectés (OpenCV): {len(corners_cv)}")
    else:
        st.error(f"Image non trouvée: {img_path}")

# ================== TAB 2: SIMPLE HOG ==================
with tab2:
    st.header("Simple HOG - Histogram of Oriented Gradients")
    st.markdown("""
    Le HOG simple calcule l'histogramme des orientations des gradients dans une région d'intérêt.
    
    Étapes:
    1. Extraire un patch au centre de l'image
    2. Calculer les gradients X et Y (Sobel)
    3. Calculer la magnitude et l'angle des gradients
    4. Créer un histogramme des orientations (8 bins)
    5. Normaliser l'histogramme
    """)
    
    hog_test_dir = "images/hog_test"
    hog_images = [
        ("circle.jpg", "Cercle"),
        ("diag.jpg", "Diagonal"),
        ("horiz.jpg", "Horizontal"),
        ("vert.jpg", "Vertical")
    ]
    
    st.subheader("Résultats HOG pour différents motifs")
    
    cols = st.columns(2)
    
    for idx, (img_name, label) in enumerate(hog_images):
        img_path = os.path.join(hog_test_dir, img_name)
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Center patch
            cy, cx = img.shape[0] // 2, img.shape[1] // 2
            center_size = 11
            half = center_size // 2
            patch = img[cy - half:cy + half + 1, cx - half:cx + half + 1]
            
            # Compute gradients
            grad_x = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=11)
            grad_y = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=11)
            magnitude = cv2.magnitude(grad_x, grad_y)
            angle = cv2.phase(grad_x, grad_y)
            
            # Create histogram
            bin_edges = np.linspace(0, 2 * np.pi, 8 + 1)
            hist, bins = np.histogram(angle, bins=bin_edges, weights=magnitude)
            hist = hist / (hist.sum() + 1e-8)
            
            col = cols[idx % 2]
            
            with col:
                st.subheader(f"Pattern: {label}")
                
                # Display image with patch highlight
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                axes[0].imshow(img, cmap='gray')
                rect = plt.Rectangle((cx - half, cy - half), 2 * half + 1, 2 * half + 1, 
                                     fill=False, edgecolor='red', linewidth=2)
                axes[0].add_patch(rect)
                axes[0].set_title(f"{label} - Patch central en rouge")
                axes[0].axis('off')
                
                # HOG histogram
                angles = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
                axes[1].bar(range(len(hist)), hist, color='steelblue', alpha=0.7)
                axes[1].set_xticks(range(len(hist)))
                axes[1].set_xticklabels([f"{int(np.degrees(a))}°" for a in angles], rotation=45)
                axes[1].set_ylabel("Magnitude normalisée")
                axes[1].set_title("HOG Histogram")
                axes[1].grid(alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.write(f"**Histogram values:** {[f'{v:.4f}' for v in hist]}")

# ================== TAB 3: IMAGE RETRIEVAL ==================
with tab3:
    st.header("Image Retrieval System")
    st.markdown("""
    Système de récupération d'images basé sur les descripteurs SIFT.
    
    Fonctionnement:
    1. Charger une image de test
    2. Extraire les descripteurs SIFT
    3. Comparer avec les images de la base de données
    4. Afficher les meilleures correspondances
    """)
    
    db_path = "images/db"
    
    if os.path.exists(db_path):
        train_path = os.path.join(db_path, "train")
        test_path = os.path.join(db_path, "test")
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            st.subheader("Structure de la base de données")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Images d'entraînement (train):**")
                train_classes = os.listdir(train_path)
                for class_name in train_classes:
                    if class_name.startswith('.'):
                        continue
                    class_path = os.path.join(train_path, class_name)
                    if os.path.isdir(class_path):
                        num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                        st.write(f"- **{class_name}**: {num_images} images")
            
            with col2:
                st.write("**Images de test (test):**")
                test_classes = os.listdir(test_path)
                for class_name in test_classes:
                    if class_name.startswith('.'):
                        continue
                    class_path = os.path.join(test_path, class_name)
                    if os.path.isdir(class_path):
                        num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                        st.write(f"- **{class_name}**: {num_images} images")
            
            st.info("""
            **Note:** Pour une implémentation complète du système de récupération, 
            il faut extraire les descripteurs SIFT/ORB de toutes les images et 
            utiliser un algorithme de matching (BruteForceMatcher ou FLANN) pour 
            trouver les images les plus similaires.
            """)
            
            # Display sample images
            st.subheader("Exemples d'images de la base de données")
            
            for class_name in train_classes[:2]:  # Show first 2 classes
                if class_name.startswith('.'):
                    continue
                class_path = os.path.join(train_path, class_name)
                if os.path.isdir(class_path):
                    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:3]
                    
                    if images:
                        st.write(f"**Classe: {class_name}**")
                        cols = st.columns(len(images))
                        for col, img_name in zip(cols, images):
                            img_path = os.path.join(class_path, img_name)
                            try:
                                img = cv2.imread(img_path)
                                if img is not None:
                                    with col:
                                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                                                use_container_width =True, 
                                                caption=img_name)
                            except:
                                pass

# Footer
st.markdown("---")
st.markdown("""
**Auteur:** Yann  
**Classe:** Learning From Images - Semestre 2  
**Technologie:** Python, OpenCV, Streamlit
""")
