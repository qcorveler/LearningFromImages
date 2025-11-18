import cv2
import glob
import numpy as np
from queue import PriorityQueue
import os

############################################################
#
#              Simple Image Retrieval
#
############################################################

test_image_path = "./images/db/test/flower2.jpg"
filename = os.path.splitext(os.path.basename(test_image_path))[0]
test_image = cv2.imread(test_image_path)
gray_test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

# show Image
def showImage(image, name):
    cv2.imshow(name, image)
    cv2.waitKey(0)


# implement distance function
def distance(a, b):
    return float(np.linalg.norm(a - b))


def create_keypoints(w, h, step=16, keypointSize=11):
    keypoints = []
    # please sample the image uniformly in a grid
    # find the keypoint size and number of sample points
    # as hyperparameters
    for x in range(0, w, step): 
        for y in range(0, h, step):
            keypoints.append(cv2.KeyPoint(x, y, keypointSize))

    return keypoints


# 1. preprocessing and load
image_paths = glob.glob('./images/db/train/*/*.jpg')
images = []
gray_images = []
for path in image_paths :
    images.append(cv2.imread(path))
    gray_images.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

# 2. create keypoints on a regular grid (cv2.KeyPoint(r, c, keypointSize), as keypoint size use e.g. 11)
descriptors = []
keypoints = create_keypoints(256, 256)

# 3. use the keypoints for each image and compute SIFT descriptors
#    for each keypoint. this compute one descriptor for each image.

sift = cv2.SIFT_create()

for image in gray_images :
    _, descriptor = sift.compute(image, keypoints)
    descriptors.append(descriptor)

# 4. use one of the query input image to query the 'image database' that
#    now compress to a single area. Therefore extract the descriptor and
#    compare the descriptor to each image in the database using the L2-norm
#    and save the result into a priority queue (q = PriorityQueue())

q = PriorityQueue()

_, test_descriptor = sift.compute(gray_test_image, keypoints)

for i, descriptor in enumerate(descriptors) : 
    dist = distance(test_descriptor, descriptor)
    q.put((dist, images[i]))

# best_match = q.get()

# showImage(test_image, "query_image")
# showImage(best_match[1], "best_match")

# 5. output (save and/or display) the query results in the order of smallest distance

# Get the 20 best matches
best_results = []
for _ in range(20):
    if not q.empty():
        best_results.append(q.get())

mini_size = (128, 128)
minis = []

for dist, img in best_results:
    mini = cv2.resize(img, mini_size)
    minis.append(mini)

# Build the grid: 2 rows, 10 columns
row1 = np.hstack(minis[:10])
row2 = np.hstack(minis[10:20])

grid = np.vstack([row1, row2])

mini_query = cv2.resize(test_image, (256, 256))
# Add the query image at the beginning
grid = np.hstack([mini_query, grid])

cv2.imwrite(f"./images/db/result/image_retrieval_results_for_{filename}.jpg", grid)

# Show the final grid
cv2.imshow(f"Top 20 Matches ordered from closest (0,0) to farthest (19,1) for {filename}", grid)
cv2.waitKey(0)
