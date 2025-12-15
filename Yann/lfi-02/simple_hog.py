import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plot_histogram(hist, bins):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()


def compute_simple_hog(img_path, plot_hist = False):

    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Center keypoints : 
    cx, cy = img.shape[0]//2, img.shape[1]//2
    
    # compute x and y gradients (sobel kernel size 11)
    center_size = 11
    half = center_size // 2
    patch = img[cy-half:cy+half+1, cx-half:cx+half+1]

    grad_x = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=11)
    grad_y = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=11)
    magnitude = cv2.magnitude(grad_x, grad_y)
    angle     = cv2.phase(grad_x, grad_y)

    # Create histogram of angles (8 bins)
    bin_edges = np.linspace(0, 2*np.pi, 8 + 1)
    hist, bins = np.histogram(angle, bins=bin_edges, weights=magnitude)

    # Normalize histogram
    hist = hist / (hist.sum() + 1e-8)
    return hist, bins, img


script_dir = os.path.dirname(os.path.abspath(__file__))
L_img_path = [
    os.path.join(script_dir, "images", "hog_test", "circle.jpg"),
    os.path.join(script_dir, "images", "hog_test", "diag.jpg"),  # <-- attention au nom !
    os.path.join(script_dir, "images", "hog_test", "horiz.jpg"),
    os.path.join(script_dir, "images", "hog_test", "vert.jpg")
]

# Création des subplots
fig, axes = plt.subplots(2, len(L_img_path), figsize=(4*len(L_img_path), 6))

for i, img_path in enumerate(L_img_path):
    hist, bins, img = compute_simple_hog(img_path)

    # Top row : image
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title(os.path.basename(img_path))

    # Bottom row : histogram
    width = 0.7 * (bins[1]-bins[0])
    center = (bins[:-1] + bins[1:])/2
    axes[1, i].bar(center, hist, align='center', width=width)
    axes[1, i].set_xlim([0,2*np.pi])
    axes[1, i].set_xlabel('Angle (rad)')
    axes[1, i].set_ylabel('Magnitude')
    axes[1, i].set_title('HOG Histogram')

plt.tight_layout()
os.makedirs("results/hog_results", exist_ok=True)
plt.savefig("results/hog_results/plot.png", dpi=300)
plt.show()
