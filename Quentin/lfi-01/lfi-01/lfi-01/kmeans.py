import numpy as np
import cv2
from typing import Tuple

############################################################
#
#                       KMEANS
#
############################################################

# k-means works in 3 steps
# 1. initialize
# 2. assign each data element to current mean (cluster center)
# 3. update mean
# then iterate between 2 and 3 until convergence, i.e. until ~smaller than 5% change rate in the overall distance or cluster centers positions 

def initialize_clusters(img: np.ndarray, num_clusters: int) -> np.ndarray:
    """
    Initialize cluster centers by randomly selecting pixels from the image.
    
    :param img (np.ndarray): The image array.
    :param num_clusters (int): The number of clusters to initialize.
    :return np.ndarray: Array of initial cluster centers.
    """
    
    # YOUR CODE HERE  
    # HINT: you load your images in uint8 format. convert your initial centers to float32 -> initial_centers.astype(np.float32)

    ## NOTE !!!!!!!!!
    ## To get full points you - ADDITIONALLY - have to develop your own init method. Please read the assignment!
    ## It should work with both init methods.

    # Reshape image to a list of pixels
    pixels = img.reshape(-1, 3)
    pixels = np.float32(pixels)

    # Select num_cluster random indices in the pixel list  
    random_idx = np.random.choice(len(pixels), num_clusters, replace=False)
    random_centers = pixels[random_idx]

    return random_centers


def assign_clusters(img: np.ndarray, cluster_centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Assign each pixel in the image to the nearest cluster center and calculate the overall distance.

    :param img (np.ndarray): The image array.
    :param cluster_centers (np.ndarray): Current cluster centers.   
    :return Tuple[np.ndarray, np.ndarray, float]: Tuple of the updated image, cluster mask, and overall distance.
    """

    # YOUR CODE HERE  
    # HINT: 
    # 1. compute distances per pixel
    # 2. find closest cluster center for each pixel
    # 3. based on new cluster centers for each pixel, create new image with updated colors (updated_img)
    # 4. compute overall distance just to print it in each step and see that we minimize here
    # you return updated_img.astype(np.uint8), closest_clusters, overall_distance
    # the updated_img is converted back to uint8 just for display reasons

    pixels = np.float32(img.reshape(-1, 3))

    # Compute the distances between each pixel and each cluster center
    num_pixels = pixels.shape[0]
    num_clusters = cluster_centers.shape[0]
    distances = np.zeros((num_pixels, num_clusters), dtype=np.float32)

    for k in range(num_clusters):
        # Pour le cluster k, on calcule la différence entre chaque pixel et le centre
        diff = pixels - cluster_centers[k]  # (N, 3)
        
        # Puis on calcule la distance euclidienne
        distances[:, k] = np.sqrt(np.sum(diff ** 2, axis=1))

    # Trouver le centre le plus proche pour chaque pixel
    closest_clusters = np.argmin(distances, axis=1)

    # Mettre à jour l'image avec les couleurs du cluster assigné
    updated_pixels = cluster_centers[closest_clusters]

    # Calculer la distance totale (pour affichage ou convergence)
    overall_distance = np.sum(np.min(distances, axis=1))

    # Reformater pour reconstruire une image
    updated_img = updated_pixels.reshape(img.shape)

    return updated_img.astype(np.uint8), closest_clusters, overall_distance



def update_cluster_centers(img: np.ndarray, cluster_assignments: np.ndarray, num_clusters: int) -> np.ndarray:
    """
    Update cluster centers as the mean of assigned pixels.

    :param img (np.ndarray): The image array.
    :param cluster_assignments (np.ndarray): Cluster assignments for each pixel.
    :param num_clusters (int): Number of clusters.
    :return np.ndarray: Updated cluster centers.
    """
    
    # YOUR CODE HERE  
    # HINT: Find the new mean for each center and return new_centers (those are new RGB colors)
    pixels = np.float32(img.reshape(-1, 3))
    new_centers = np.zeros((num_clusters, 3), dtype=np.float32)

    for k in range(num_clusters):
        assigned_pixels = pixels[cluster_assignments == k]
        if len(assigned_pixels) > 0:
            new_centers[k] = np.mean(assigned_pixels, axis=0)
        else:
            # Si un cluster est vide, on réinitialise à un pixel aléatoire
            new_centers[k] = pixels[np.random.randint(0, len(pixels))]

    return new_centers

def kmeans_clustering(img: np.ndarray, num_clusters: int = 3, max_iterations: int = 100, tolerance: float = 0.01) -> np.ndarray:
    """
    Apply K-means clustering to do color quantization. Main k-means function iterating over max_iterations and stopping if
    the error rate of change is less then 2% for consecutive iterations, i.e. the
    algorithm converges, centers don't change in between iterations anymore. 
    
    :param img (np.ndarray): The image to be segmented.
    :param num_clusters (int): The number of clusters.
    :param max_iterations (int): The maximum number of iterations.
    :param tolerance (float): The convergence tolerance.
    :return np.ndarray: The segmented image.
    """
    
    # YOUR CODE HERE  
    # initialize the clusters
    # for loop over max_iterations
    # in each loop
    # 1. assign clusters, this gives you a quantized image
    # 2. update cluster centers
    # 3. check for early break with tolerance
    # return updated_img

    cluster_centers = initialize_clusters(img, num_clusters)

    last_distance = None
    for i in range(max_iterations):
        # 1. Assign pixels to nearest cluster
        updated_img, cluster_assignments, overall_distance = assign_clusters(img, cluster_centers)

        # 2. Update cluster centers
        new_centers = update_cluster_centers(img, cluster_assignments, num_clusters)

        # 3. Check convergence
        if last_distance is not None:
            change = abs(last_distance - overall_distance) / last_distance
            print(f"Iteration {i+1}: distance={overall_distance:.2f}, change={change*100:.2f}%")
            if change < tolerance:
                print("Converged!")
                break
        else:
            print(f"Iteration {i+1}: distance={overall_distance:.2f}")

        cluster_centers = new_centers
        last_distance = overall_distance

    return updated_img

def kmeans_clustering_lab(img: np.ndarray, num_clusters: int = 3, max_iterations: int = 100, tolerance: float = 0.01):
    LabImage = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    cluster_centers = initialize_clusters(LabImage, num_clusters)

    last_distance = None
    for i in range(max_iterations):
        # 1. Assign pixels to nearest cluster
        updated_img, cluster_assignments, overall_distance = assign_clusters(LabImage, cluster_centers)

        # 2. Update cluster centers
        new_centers = update_cluster_centers(LabImage, cluster_assignments, num_clusters)

        # 3. Check convergence
        if last_distance is not None:
            change = abs(last_distance - overall_distance) / last_distance
            print(f"Iteration {i+1}: distance={overall_distance:.2f}, change={change*100:.2f}%")
            if change < tolerance:
                print("Converged!")
                break
        else:
            print(f"Iteration {i+1}: distance={overall_distance:.2f}")

        cluster_centers = new_centers
        last_distance = overall_distance

    updated_img = np.clip(updated_img, 0, 255).astype(np.uint8)
    updated_img = cv2.cvtColor(updated_img, cv2.COLOR_LAB2BGR)
    return updated_img


def load_and_process_image(file_path: str, scaling_factor: float = 0.5) -> np.ndarray:
    """
    Load and preprocess an image.
    
    :param file_path (str): Path to the image file.
    :param scaling_factor (float): Scaling factor to resize the image.        
    :return np.ndarray: The preprocessed image.
    """
    image = cv2.imread(file_path)

    # Note: the scaling helps to do faster computation :) 
    image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return image

def main():
    file_path = './graffiti.png'
    num_clusters = 64
    
    img = load_and_process_image(file_path, 1)
    segmented_img = kmeans_clustering(img, num_clusters)
    segmented_imgLAB = kmeans_clustering_lab(img, num_clusters)
    
    cv2.imshow("Original", img)
    cv2.imshow("Color-based Segmentation Kmeans-Clustering", segmented_img)
    cv2.imshow("Color-based Segmentation Kmeans-Clustering With LAB conversion", segmented_imgLAB)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
