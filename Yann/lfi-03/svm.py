import numpy as np
import cv2
import glob
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 


############################################################
#
#              Support Vector Machine
#              Image Classification
#
############################################################

img_db_train_path = './images/db/train/*/*.jpg'
img_db_test_path = './images/db/test/*.jpg'

# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use ~15x15 keypoints on each image with subwindow of 21px (diameter)

# For of all let assure that all train and test image have the same size
img_size = None
L_non_same_size_path = []
for img_path in glob.glob(img_db_train_path+'') + glob.glob(img_db_test_path+''):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_size == None :
        img_size = img.shape
    if img_size != img.shape:
        L_non_same_size_path.append(img_path)
        print("Img_path", img_path)
    # print(img_path, img.shape)
# Only one img does not fit the size it's (255,256) instead of (256,256) so lets fix it
for img_path in L_non_same_size_path:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (img_size[1], img_size[0]))
    cv2.imwrite(img_path, img_resized)

# Now we can extract the same number of keypoints for each image
# Let's create them
def create_keypoints(w,h):
    grid_size = 15
    subwindow_size = 21
    keypoints = []
    step_w = int(np.round(w / grid_size))
    step_h = int(np.round(h / grid_size))
    keypointSize = float(subwindow_size)
    step = min(step_w, step_h)
    start_x = step // 2
    start_y = step // 2
    for y in range(start_y, h - start_y + 1, step):
        for x in range(start_x, w - start_x + 1, step):
            # Créer l'objet KeyPoint : (x, y, size)
            keypoints.append(cv2.KeyPoint(float(x), float(y), keypointSize))
    return keypoints

# Let's plot one img to see where the keypoints are
img = cv2.imread('images/db/train/cars/1137646735_2fb2752249.jpg', cv2.IMREAD_GRAYSCALE)

h, w = img.shape
keypoints = create_keypoints(w, h)
img_kp = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Keypoints location', img_kp)
cv2.waitKey(2000)
cv2.destroyAllWindows()


# Now let compute SIFT extraction for all training images
sift = cv2.SIFT_create()
X_train = []
y_train = []


# 2. each descriptor (set of features) need to be flattened in one vector
# That means you need a X_train matrix containing a shape of (num_train_images, num_keypoints*num_entry_per_keypoint)
# num_entry_per_keypoint = histogram orientations as talked about in class
# You also need a y_train vector containing the labels encoded as integers
for img_path in glob.glob(img_db_train_path):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    current_keypoints = create_keypoints(img.shape[1], img.shape[0])
    len_kp = len(current_keypoints)
    # Descriptor in keypoints
    _, descriptors = sift.compute(img, current_keypoints) 
    flattened_descriptor = descriptors.flatten()
    
    X_train.append(flattened_descriptor)
    class_name = img_path.split('\\')[-2]
    y_train.append(class_name)
# print(X_train)
# print(y_train)
print("-"*20 + " X_train building " + "-"*20)
print(f"Size of X_train: {len(X_train)} x {len(X_train[0])}")
print(f"Where each descriptor has size: {len(X_train[0])} (num_keypoints x num_entry_per_keypoint)")

print(f"Number of keypoints per image : {len_kp}")
print(f"And number of entries per keypoint : 128 (SIFT descriptor size)\n\n\n")

# We need to create the same for test images :
X_test = []
y_test = []
for img_path in glob.glob(img_db_test_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    current_keypoints = create_keypoints(img.shape[1], img.shape[0])
    # Descriptor in keypoints
    _, descriptors = sift.compute(img, current_keypoints) 
    flattened_descriptor = descriptors.flatten()
    
    X_test.append(flattened_descriptor)
y_test = ["cars", "faces", "flowers", "flowers"]
# 3. We use scikit-learn to train a SVM classifier - however you need to test with different kernel options to get
# good results for our dataset.
# 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
# the same way we did for the training (except for a single image now) and use .predict()
# to classify the image
kernels = ['linear', 'poly', 'rbf']
dict_y_pred = {}
for kernel in kernels:
    clf = svm.SVC(kernel=kernel)  
    clf.fit(X_train, y_train)
    # Now we can predict the test set
    y_pred = clf.predict(X_test)
    dict_y_pred[kernel] = y_pred

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matrice de Confusion (Kernel={clf.kernel})")
    plt.show()    


# 5. output the class + corresponding name
print("-"*20 + " SVM results " + "-"*20)
print("True labels for test set:", y_test)
print(f"Predictions for different kernels: {dict_y_pred}")


