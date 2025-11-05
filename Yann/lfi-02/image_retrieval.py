import numpy as np

import cv2
import glob
from queue import PriorityQueue
import os
############################################################
#
#              Simple Image Retrieval
#
############################################################

# --- Config ---
db_path_train = "images/db/train"
db_path_test = "images/db/test"
# img_ext = ".jpg"

# def load_data_base(db_path = db_path_train):
    # images = {}
    # for class_name in os.listdir(db_path):
    #     class_path = os.path.join(db_path, class_name)
    #     images[class_name] = []
    #     for img_name in os.listdir(class_path):
    #         img_path = os.path.join(class_path, img_name)
    #         img = cv2.imread(img_path)
    #         images[class_name].append(img)
    #     # print(len(images[class_name]) == len(os.listdir(class_path)))
    # return images

# implement distance function
def distance(a, b):
    # YOUR CODE HERE
    pass


def create_keypoints(w, h):
    keypoints = []
    keypointSize = 11
    # please sample the image uniformly in a grid
    # find the keypoint size and number of sample points
    # as hyperparameters

    keypoints = []
    step = 16
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            keypoints.append(cv2.KeyPoint(float(x), float(y), keypointSize))
    return keypoints


# 1. preprocessing and load
images = glob.glob('./images/db/train/*/*.jpg')


# 2. create keypoints on a regular grid (cv2.KeyPoint(r, c, keypointSize), as keypoint size use e.g. 11)
descriptors = []
keypoints = create_keypoints(256, 256)

# 3. use the keypoints for each image and compute SIFT descriptors
#    for each keypoint. this compute one descriptor for each image.


# YOUR CODE HERE

# 4. use one of the query input image to query the 'image database' that
#    now compress to a single area. Therefore extract the descriptor and
#    compare the descriptor to each image in the database using the L2-norm

#    and save the result into a priority queue (q = PriorityQueue())

# YOUR CODE HERE

# 5. output (save and/or display) the query results in the order of smallest distance

# YOUR CODE HERE


if __name__ == "__main__" :
    img = cv2.imread(images[np.random.randint(len(images))])
    cv2.imshow("A random image", img)
    cv2.waitKey(0)  # attend une touche
    cv2.destroyAllWindows()  # ferme la fenêtre
