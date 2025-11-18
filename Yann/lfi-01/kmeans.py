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

def initialize_clusters(img: np.ndarray, num_clusters: int, n_init:int = 1) -> np.ndarray:
    """
    Initialize cluster centers by randomly selecting pixels from the image.
    
    :param img (np.ndarray): The image array.
    :param num_clusters (int): The number of clusters to initialize.
    :return np.ndarray: Array of initial cluster centers.
    """
    
    # YOUR CODE HERE  
    # HINT: you load your images in uint8 format. convert your initial centers to float32 -> initial_centers.astype(np.float32)
    h, w = img.shape[:2]
    nb_pix = h * w
    # For mor intialisation test :
    best_centers = None
    best_spread = -np.inf

    for _ in range(n_init):    
        pos_pix = [nb_pix//num_clusters * i  + nb_pix//num_clusters//2  for i in range(num_clusters)]
        coords = [(pos//w , np.random.randint(0,w)) for pos in pos_pix]
        first_centers = np.array([img[row, col] for row, col in coords], dtype=np.float32)

        # We keep the bigger distance
        distances = np.linalg.norm(first_centers[:, None, :] - first_centers[None, :, :], axis=2)
        avg_dist = np.mean(distances)
        if avg_dist > best_spread:
            best_spread = avg_dist
            best_centers = first_centers
    ## NOTE !!!!!!!!!
    ## To get full points you - ADDITIONALLY - have to develop your own init method. Please read the assignment!
    ## It should work with both init methods.
    return best_centers.astype(np.float32)


def assign_clusters(img: np.ndarray, cluster_centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Assign each pixel in the image to the nearest cluster center and calculate the overall distance.

    :param img (np.ndarray): The image array.
    :param cluster_centers (np.ndarray): Current cluster centers.   
    :return Tuple[np.ndarray, np.ndarray, float]: Tuple of the updated image, cluster mask, and overall distance.
    """
    

    # YOUR CODE HERE  
    # HINT:
    pixels = np.float32(img.reshape(-1, 3))
    n_clus , n_pix = len(cluster_centers), len(pixels)
    # for pix in pixels:
    #     dist_pix = []
    #     dist_min= +np.inf
    #     for cluster in cluster_centers:
    #         dist = np.linalg.norm(pix - cluster, axis=2)
    #         if dist < dist_min:
    #             clust_pix = cluster
          
        
    # 1. compute distances per pixel
    pixel_values = img.reshape((-1, img.shape[-1])).astype(np.float32)  # shape (num_pixels, 3)
    distances = np.linalg.norm(pixel_values[:, np.newaxis] - cluster_centers, axis=2)  # shape (num_pixels, num_clusters)
    
    # 2. find closest cluster center for each pixel
    closest_clusters = np.argmin(distances, axis=1)
    # 3. based on new cluster centers for each pixel, create new image with updated colors (updated_img)
    updated_img = cluster_centers[closest_clusters]
    updated_img = updated_img.reshape(img.shape)
    # 4. compute overall distance just to print it in each step and see that we minimize here
    overall_distance = np.sum(np.min(distances, axis=1))
    print(overall_distance)
    # you return updated_img.astype(np.uint8), closest_clusters, overall_distance
    return updated_img.astype(np.uint8), closest_clusters, overall_distance
    # the updated_img is converted back to uint8 just for display reasons


def update_cluster_centers(img: np.ndarray, cluster_assignments: np.ndarray, num_clusters: int) -> np.ndarray:
    """
    Update cluster centers as the mean of assigned pixels.

    :param img (np.ndarray): The image array.
    :param cluster_assignments (np.ndarray): Cluster assignments for each pixel.
    :param num_clusters (int): Number of clusters.
    :return np.ndarray: Updated cluster centers.
    """
    pixel_values = img.reshape((-1, img.shape[-1]))

    # Initialize an array to store the new cluster centers
    new_centers = np.zeros((num_clusters, img.shape[-1]))
    for i in range(num_clusters):
        cluster_pixels = pixel_values[cluster_assignments == i]

        # Compute the mean of these pixels
        if len(cluster_pixels) > 0:
            new_centers[i] = np.mean(cluster_pixels, axis=0)
        else:
            new_centers[i] = np.zeros_like(new_centers[i])

    return new_centers
    # YOUR CODE HERE  
    # HINT: Find the new mean for each center and return new_centers (those are new RGB colors)

def kmeans_clustering(img: np.ndarray, num_clusters: int = 3, max_iterations: int = 100, tolerance: float = 0.01, n_init: int = 1) -> np.ndarray:
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
    cluster_centers = initialize_clusters(img, num_clusters, n_init=n_init)
    # for loop over max_iterations
    prev_distance = float('inf')
    # in each loop
    for iteration in range(max_iterations):
    # 1. assign clusters, this gives you a quantized image
        updated_img, cluster_assignments, overall_distance = assign_clusters(img, cluster_centers)
    # 2. update cluster centers
        new_centers = update_cluster_centers(img, cluster_assignments, num_clusters)
    # 3. check for early break with tolerance
        if abs(prev_distance - overall_distance) / prev_distance < tolerance:
            print(f"Converged at iteration {iteration} with distance {overall_distance}.")
            break
    # return updated_img
        prev_distance = overall_distance
        cluster_centers = new_centers

    return updated_img

def load_and_process_image(file_path: str, scaling_factor: float = 0.5, blur_it: bool = False, lab_it: bool = False) -> np.ndarray:
    """
    Load and preprocess an image.
    
    :param file_path (str): Path to the image file.
    :param scaling_factor (float): Scaling factor to resize the image.        
    :return np.ndarray: The preprocessed image.
    """
    img = cv2.imread(file_path)
    if blur_it:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if lab_it:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


    
    # Note: the scaling helps to do faster computation :) 
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return img

def main():
    file_path = 'graffiti.png'
    num_clusters = [4, 8, 16, 20] 
    img = load_and_process_image(file_path)
    print(img.shape)
    segmented_images = img
    for k in num_clusters:
        # img = load_and_process_image(file_path)
        # segmented_img = kmeans_clustering(img, k, n_init=1)
        # segmented_images = np.concatenate((segmented_images, segmented_img), axis = 1)
        img = load_and_process_image(file_path, blur_it=True, lab_it=True)
        segmented_img = kmeans_clustering(img, k, n_init=20)
        segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_LAB2BGR)
        segmented_images = np.concatenate((segmented_images, segmented_img), axis = 1)
    cv2.imshow("Color-based Segmentation Kmeans-Clustering : Normal, first implentation, upgraded implementation", segmented_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
