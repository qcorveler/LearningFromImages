import numpy as np

import cv2
import glob
from queue import PriorityQueue
import os
import pickle
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
    assert len(a) == len(b)
    dist = 0
    for i in range(len(a)) :

        ai , bi = a[i], b[i]
        dist += (ai-bi)*(ai-bi)
    return np.sqrt(dist)


def create_keypoints(w, h, step = 16):
    keypoints = []
    keypointSize = 11
    # please sample the image uniformly in a grid
    # find the keypoint size and number of sample points
    # as hyperparameters
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
def calculate_sift(img) : 
    w, h = img.shape[:2]
    keypoints = create_keypoints(w, h)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(img, keypoints)
    return descriptors
img = cv2.imread(images[np.random.randint(len(images))])
descriptors = calculate_sift(img)


descrip_path = "images/db/descriptors_full.p"
if not os.path.exists(descrip_path):
    print("We compute descriptors")
    descriptors_full = {}
    for img_path in images : 
        img = cv2.imread(img_path)
        descriptors_full[img_path] = calculate_sift(img)
    with open(descrip_path, "wb") as f:
        pickle.dump(descriptors_full, f)
else : 
    print("We load descriptors")
    with open(descrip_path, "rb") as f:
        descriptors_full = pickle.load(f)
# 4. use one of the query input image to query the 'image database' that
#    now compress to a single area. Therefore extract the descriptor and
#    compare the descriptor to each image in the database using the L2-norm
images_test = glob.glob('./images/db/test/*.jpg')
distance_per_imgtest_forimgtrain = {}
if not os.path.exists('images/db/result/dict_results_dist.p'):
    for img_path in images_test : 
        img = cv2.imread(img_path)
        descriptors_test = calculate_sift(img)
        dict_dist = {}
        # Now we have the descriptors of each test images 
        # We have to compare it to each train images

        for train_path in descriptors_full.keys() : 
            descriptors = descriptors_full[train_path]
            dist_test = 0
            assert len(descriptors) == len(descriptors_test)
            for i in range(len(descriptors_test)):
                dist_test += distance(descriptors[i], descriptors_test[i])
                print(type(dist_test))
            dict_dist[train_path] = dist_test

            
        distance_per_imgtest_forimgtrain[img_path] = dict_dist
    with open('images/db/result/dict_results_dist.p', "wb") as f:
            pickle.dump(distance_per_imgtest_forimgtrain, f)
with open('images/db/result/dict_results_dist.p', "rb") as f:
    print('We load the distance between all the images')
    distance_per_imgtest_forimgtrain = pickle.load(f)


#    and save the result into a priority queue (q = PriorityQueue())
q = PriorityQueue()
for test_img, dists in distance_per_imgtest_forimgtrain.items():
    # print(distance_per_imgtest_forimgtrain)
    for train_img, dist in dists.items():
        # print(train_img, '\n')
        q.put((float(dist), (test_img, train_img)))
print(q)

# 5. output (save and/or display) the query results in the order of smallest distance

# YOUR CODE HERE


# if __name__ == "__main__" :
#     print(images_test)
    # img = cv2.imread(images[np.random.randint(len(images))])
    # descriptors = calculate_sift(img)
    # print(descriptors)
    # print(f"Le nombre de descripteur est de : {len(descriptors)}")
    # cv2.imshow("A random image", img)
    # cv2.waitKey(0)  # attend une touche
    # cv2.destroyAllWindows()  # ferme la fenêtre
