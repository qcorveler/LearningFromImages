import numpy as np
import cv2
import glob
from sklearn import svm
# import Levenshtein


############################################################
#
#              Support Vector Machine
#              Image Classification
#
############################################################


# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use ~15x15 keypoints on each image with subwindow of 21px (diameter)

def create_sift_descriptor(image, keypoints):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(image, keypoints)
    return descriptors

def create_keypoints(w, h, step=16, keypointSize=21):
    keypoints = []
    for x in range(0, w, step): 
        for y in range(0, h, step):
            keypoints.append(cv2.KeyPoint(x, y, keypointSize))

    return keypoints



# 2. each descriptor (set of features) need to be flattened in one vector
# That means you need a X_train matrix containing a shape of (num_train_images, num_keypoints*num_entry_per_keypoint)
# num_entry_per_keypoint = histogram orientations as talked about in class
# You also need a y_train vector containing the labels encoded as integers

# Load training images and extract SIFT descriptors
image_paths = glob.glob('./images/db/train/*/*.jpg')
X_train = []
y_train = []
label_map = {}
label_map_inversed = {}
label_counter = 0

for image_path in image_paths:
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape

    # Create keypoints
    keypoints = create_keypoints(w, h)

    # Extract SIFT descriptors
    descriptors = create_sift_descriptor(image, keypoints)

    # Flatten descriptors and add to training data
    X_train.append(descriptors.flatten())

    # Extract label from path and encode as integer
    label = image_path.split('\\')[-2]
    if label not in label_map:
        label_map[label] = label_counter
        label_map_inversed[label_counter] = label
        label_counter += 1
    print(label_map[label])    
    print(label)
    y_train.append(label_map[label])

# 3. We use scikit-learn to train a SVM classifier - however you need to test with different kernel options to get
# good results for our dataset.

classifier_linear = svm.SVC(kernel='linear')

classifier_linear.fit(X_train, y_train)

classifier_rbf = svm.SVC(kernel="rbf")
classifier_rbf.fit(X_train, y_train)

classifier_poly = svm.SVC(kernel='poly')
classifier_poly.fit(X_train, y_train)

# 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
# the same way we did for the training (except for a single image now) and use .predict()
# to classify the image
X_test = []
y_test = []
test_image_paths = glob.glob('./images/db/test/*.jpg')
for image_path in test_image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape

    # Create keypoints
    keypoints = create_keypoints(w, h)

    # Extract SIFT descriptors
    descriptors = create_sift_descriptor(image, keypoints)
    X_test.append(descriptors.flatten())

    # label test data
    image_name = image_path.split('/')[-1][:-4]
    if 'car' in image_name :
        y_test.append(label_map['cars'])
    elif 'flower' in image_name :
        y_test.append(label_map['flowers'])
    elif 'face' in image_name :
        y_test.append(label_map['faces']) 

# prediction
y_pred_linear = classifier_linear.predict(X_test)
y_pred_rbf = classifier_rbf.predict(X_test)
y_pred_poly = classifier_poly.predict(X_test)

# 5. output the class + corresponding name

print("Linear kernel classifier")
errors = 0
for i in range(len(test_image_paths)) :
    if y_pred_linear[i] != y_test[i]:
        errors += 1
    print(f"{test_image_paths[i].split('/')[-1]} : \tprediction = {y_pred_linear[i]} ({label_map_inversed[y_pred_linear[i]]}),   \treality = {y_test[i]} ({label_map_inversed[y_test[i]]})")

accuracy = ((len(test_image_paths) - errors) / (len(test_image_paths)))*100
print(f"accuracy on test data : {accuracy}%")

print('---')
print("RBF kernel classifier")
errors = 0
for i in range(len(test_image_paths)) :
    if y_pred_rbf[i] != y_test[i]:
        errors += 1
    print(f"{test_image_paths[i].split('/')[-1]} : \tprediction = {y_pred_rbf[i]} ({label_map_inversed[y_pred_rbf[i]]}),    \treality = {y_test[i]} ({label_map_inversed[y_test[i]]})")
accuracy = ((len(test_image_paths) - errors) / (len(test_image_paths)))*100
print(f"accuracy on test data : {accuracy}%")

print('---')
print("Polynomial kernel classifier")
errors = 0
for i in range(len(test_image_paths)) :
    if y_pred_poly[i] != y_test[i]:
        errors += 1
    print(f"{test_image_paths[i].split('/')[-1]} : \tprediction = {y_pred_poly[i]} ({label_map_inversed[y_pred_poly[i]]}),    \treality = {y_test[i]} ({label_map_inversed[y_test[i]]})")

accuracy = ((len(test_image_paths) - errors) / (len(test_image_paths)))*100
print(f"accuracy on test data : {accuracy}%")