# 🖼️ Learning From Images - LFI-02 Streamlit App

Application interactive pour explorer les techniques de traitement d'images implémentées dans le cadre du cours LFI-02.

## Fonctionnalités

### 1. Harris Corner Detection
- Détection automatique des coins dans les images
- Implémentation manuelle et comparaison avec OpenCV
- Contrôle interactif du seuil de détection
- Visualisation en temps réel de la heatmap

### 2. Simple HOG (Histogram of Oriented Gradients)
- Calcul des histogrammes des gradients orientés
- Analyse de 4 motifs différents (Cercle, Diagonal, Horizontal, Vertical)
- Visualisation des patches et des histogrammes

### 3. Image Retrieval System
- Visualisation de la structure de la base de données
- Affichage des images d'entraînement et de test
- Base pour implémenter un système complet de récupération d'images

## Installation

### Prérequis
- Python 3.8+
- pip

### Étapes

1. **Installer les dépendances:**
```bash
pip install -r requirements.txt
```

2. **Lancer l'application:**
```bash
streamlit run app.py
```

3. L'application s'ouvrira automatiquement dans votre navigateur à `http://localhost:8501`

## Structure du projet

```
lfi-02/
├── app.py                      # Application Streamlit
├── requirements.txt            # Dépendances Python
├── harris.py                   # Implémentation Harris Corner Detection
├── simple_hog.py               # Implémentation Simple HOG
├── image_retrieval.py          # Implémentation Image Retrieval
├── images/
│   ├── graffiti.png           # Image pour Harris
│   ├── hog_test/              # Images pour HOG
│   │   ├── circle.jpg
│   │   ├── diag.jpg
│   │   ├── horiz.jpg
│   │   └── vert.jpg
│   └── db/                    # Base de données
│       ├── train/             # Images d'entraînement
│       │   ├── cars/
│       │   ├── faces/
│       │   └── flowers/
│       └── test/              # Images de test
└── results/                    # Résultats générés

```

## Utilisation

### Harris Corners Tab
1. Naviguez vers l'onglet "Harris Corners"
2. Utilisez le slider pour ajuster le seuil de détection
3. Comparez l'implémentation manuelle avec OpenCV

### Simple HOG Tab
1. Naviguez vers l'onglet "Simple HOG"
2. Observez les histogrammes pour différents motifs
3. Analysez comment les gradients varient selon les orientations

### Image Retrieval Tab
1. Naviguez vers l'onglet "Image Retrieval"
2. Visualisez la structure de la base de données
3. Observez les exemples d'images

## Notes techniques

- **Harris Detection:** Utilise les gradients Sobel et la matrice d'autocorrélation
- **HOG:** Calcule l'histogramme sur un patch de 11×11 pixels au centre
- **Image Retrieval:** Peut être étendu avec des descripteurs SIFT/ORB

## Améliorations futures

- [ ] Implémentation complète du système de récupération d'images avec matching
- [ ] Export des résultats en PDF
- [ ] Benchmark de performance
- [ ] Support de webcam pour test en temps réel
- [ ] Paramètres ajustables pour tous les algorithmes

## Auteur

Yann

## Date

Novembre 2025
