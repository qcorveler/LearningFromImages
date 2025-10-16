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
    num_clusters = 4
    
    img = load_and_process_image(file_path)
    segmented_img = kmeans_clustering(img, num_clusters)
    
    cv2.imshow("Original", img)
    cv2.imshow("Color-based Segmentation Kmeans-Clustering", segmented_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
