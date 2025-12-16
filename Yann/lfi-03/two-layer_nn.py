import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# set a fixed seed for reproducability
np.random.seed(0)

nn_img_size = 32
num_classes = 3
learning_rate = 0.01
num_epochs = 500
batch_size = 4

# Toggle this between 'mse' and 'crossentropy'
loss_mode = 'crossentropy'

loss_train_hist = []

##################################################
## Please implement a two layer neural network  ##
##################################################


def relu(x):
    """ReLU activation function"""
    return np.maximum(x, 0)


def relu_derivative(output):
    """derivative of the ReLU activation function"""
    output[output <= 0] = 0
    output[output > 0] = 1
    return output


def softmax(z):
    """softmax function to transform values to probabilities"""
    z -= z.max()
    z = np.exp(z)
    sum_z = z.sum(1, keepdims=True)
    return z / sum_z


def loss_mse(activation, y_batch):
    """mean squared loss function"""
    # use MSE error as loss function
    # Hint: the computed error needs to get normalized over the number of samples
    loss = ((activation - y_batch)**2).sum()
    mse = 1.0 / activation.shape[0] * loss
    return mse


def loss_crossentropy(activation, y_batch):
    """cross entropy loss function"""
    batch_size = y_batch.shape[0]
    loss = (-y_batch * np.log(activation)).sum() / batch_size
    return loss


def loss_deriv_mse(activation, y_batch):
    """derivative of the mean squared loss function"""
    dCda2 = (1 / activation.shape[0]) * (activation - y_batch)
    return dCda2


def loss_deriv_crossentropy(activation, y_batch):
    """derivative of the mean cross entropy loss function, that includes the derivate of the softmax
       for further explanations see here: https://deepnotes.io/softmax-crossentropy
    """
    batch_size = y_batch.shape[0]
    # Note: simple assignment creates a reference, potentially modifying the original activation
    # Using .copy() is safer if activation is needed later, though for this script it's okay.
    dCda2 = activation
    dCda2[range(batch_size), np.argmax(y_batch, axis=1)] -= 1
    dCda2 /= batch_size
    return dCda2


def setup_train():
    """train function"""
    # load and resize train images in three categories
    # cars = 0, flowers = 1, faces = 2 ( true_ids )
    train_images_cars = glob.glob('./images/db/train/cars/*.jpg')
    train_images_flowers = glob.glob('./images/db/train/flowers/*.jpg')
    train_images_faces = glob.glob('./images/db/train/faces/*.jpg')
    if not train_images_cars or not train_images_flowers or not train_images_faces:
        # Create dummy data if no images found to prevent crash during testing of the script structure
        print("Warning: No images found. Generating dummy data for testing code structure.")
        X_train = np.random.rand(12, nn_img_size * nn_img_size).astype(np.float32)
        y_train = np.zeros((12, num_classes))
        y_train[:4, 0] = 1
        y_train[4:8, 1] = 1
        y_train[8:, 2] = 1
        return X_train, y_train

    train_images = [train_images_cars, train_images_flowers, train_images_faces]
    num_rows = len(train_images_cars) + len(train_images_flowers) + len(
        train_images_faces)
    X_train = np.zeros((num_rows, nn_img_size * nn_img_size))
    y_train = np.zeros((num_rows, num_classes))

    counter = 0
    for (label, fnames) in enumerate(train_images):
        for fname in fnames:
            print(label, fname)
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (nn_img_size, nn_img_size),
                             interpolation=cv2.INTER_AREA)

            # fill matrices X_train - each row is an image vector
            # y_train - one-hot encoded, put only a 1 where the label is correct for the row in X_train
            y_train[counter, label] = 1
            X_train[counter] = img.flatten().astype(np.float32)

            counter += 1

    return X_train, y_train


def forward(X_batch, y_batch, W1, W2, b1, b2):
    """forward pass in the neural network """
    
    # Layer 1: Input -> Hidden
    z1 = np.dot(X_batch, W1) + b1
    a1 = relu(z1)

    # Layer 2: Hidden -> Output
    z2 = np.dot(a1, W2) + b2
    
    # Activation and Loss depending on mode
    if loss_mode == 'mse':
        # For MSE, we typically use the raw output (Linear) 
        # as calculating MSE on Softmax probabilities can lead to vanishing gradients.
        a2 = z2
        loss = loss_mse(a2, y_batch)
    elif loss_mode == 'crossentropy':
        # For CrossEntropy, we must use Softmax
        a2 = softmax(z2)
        loss = loss_crossentropy(a2, y_batch)
    else:
        raise ValueError(f"Unknown loss mode: {loss_mode}")

    # Return loss, activations, and z1 (needed for backward pass mask)
    return loss, a2, a1, z1


def backward(a2, a1, X_batch, y_batch, W2, m1):
    """backward pass in the neural network """
    # m1 is passed as z1 (pre-activation of layer 1) from the forward pass
    
    # 1. Gradient at Output (dC/dZ2)
    if loss_mode == 'mse':
        # dC/dZ2 = dC/dA2 * dA2/dZ2. 
        # Since A2 is linear (A2=Z2), derivative is 1. So dC/dZ2 = dC/dA2.
        dZ2 = loss_deriv_mse(a2, y_batch)
    elif loss_mode == 'crossentropy':
        # The provided helper computes the combined Softmax+CE derivative
        # Note: We pass a copy to avoid corrupting a2 if needed elsewhere
        dZ2 = loss_deriv_crossentropy(a2.copy(), y_batch)
        
    # 2. Gradients for Layer 2 parameters (W2, b2)
    # dC/dW2 = a1.T * dZ2
    dCdW2 = np.dot(a1.T, dZ2)
    # dC/db2 = sum(dZ2)
    dCdb2 = np.sum(dZ2, axis=0)

    # 3. Propagate error to Layer 1 (dC/dA1)
    # dC/dA1 = dZ2 * W2.T
    dA1 = np.dot(dZ2, W2.T)

    # 4. Gradient through ReLU (dC/dZ1)
    # dC/dZ1 = dC/dA1 * ReLU'(Z1)
    # m1 is z1. relu_derivative modifies in-place, so we pass a copy.
    dZ1 = dA1 * relu_derivative(m1.copy())

    # 5. Gradients for Layer 1 parameters (W1, b1)
    dCdW1 = np.dot(X_batch.T, dZ1)
    dCdb1 = np.sum(dZ1, axis=0)

    return dCdW1, dCdW2, dCdb1, dCdb2


def train(X_train, y_train):
    """ train procedure """
    h = 1500
    std = 0.001
    
    input_dim = X_train.shape[1]
    
    # initialize W1, W2, b1, b2 randomly
    W1 = std * np.random.randn(input_dim, h)
    b1 = np.zeros(h)
    W2 = std * np.random.randn(h, num_classes)
    b2 = np.zeros(num_classes)

    # run for num_epochs
    for i in range(num_epochs):

        # use only a batch of batch_size of the training images in each run
        # sample the batch images randomly from the training set
        indices = np.random.choice(X_train.shape[0], batch_size, replace=False)
        X_batch = X_train[indices]
        y_batch = y_train[indices]

        # forward pass
        loss, a2, a1, z1 = forward(X_batch, y_batch, W1, W2, b1, b2)

        # add loss to loss_train_hist for plotting
        loss_train_hist.append(loss)

        if i % 10 == 0:
            print("iteration %d: loss %f" % (i, loss))

        # backward pass
        # We pass z1 as 'm1' to calculate the ReLU derivative correctly
        dW1, dW2, db1, db2 = backward(a2, a1, X_batch, y_batch, W2, z1)

        # update weights (Gradient Descent)
        W1 -= learning_rate * dW1
        W2 -= learning_rate * dW2
        b1 -= learning_rate * db1
        b2 -= learning_rate * db2

    return W1, W2, b1, b2


X_train, y_train = setup_train()
W1, W2, b1, b2 = train(X_train, y_train)

# predict the test images, load all test images and
# run prediction by computing the forward pass
test_images = []
# Ensure these files exist or wrap in try/except blocks
try:
    test_images.append((cv2.imread('./images/db/test/flower2.jpg',
                                   cv2.IMREAD_GRAYSCALE), 1))
    test_images.append((cv2.imread('./images/db/test/car.jpg',
                                   cv2.IMREAD_GRAYSCALE), 0))
    test_images.append((cv2.imread('./images/db/test/face.jpg',
                                   cv2.IMREAD_GRAYSCALE), 2))
except Exception as e:
    print("Error loading test images. Make sure paths are correct.")

print("------------------------------------")
print(f"Test Results (Loss Mode: {loss_mode})")

for ti in test_images:
    if ti[0] is None:
        continue
    
    resized_ti = cv2.resize(ti[0], (nn_img_size, nn_img_size),
                            interpolation=cv2.INTER_AREA)
    x_test = resized_ti.reshape(1, -1).astype(np.float32)
    
    # Manual forward pass for prediction
    # Layer 1
    z1_test = np.dot(x_test, W1) + b1
    a1_test = relu(z1_test)
    
    # Layer 2
    z2_test = np.dot(a1_test, W2) + b2
    
    # Output
    if loss_mode == 'mse':
        a2_test = z2_test
    else:
        a2_test = softmax(z2_test)
        
    print(f"Test output - values: {a2_test} \t pred_id: {np.argmax(a2_test)} \t true_id: {ti[1]}")

# print("------------------------------------")
# print("Test model output Weights:", W1, W2)
# print("Test model output bias:", b1, b2)

plt.title(f"Training Loss vs. Number of Training Epochs ({loss_mode})")
plt.xlabel("Training Epochs")
plt.ylabel("Training Loss")
plt.plot(range(1, num_epochs + 1), loss_train_hist, label="Train")
if loss_mode == 'mse':
    plt.ylim((0, 3.))
else:
    plt.ylim((0, 3.)) # Adjust scale for Cross Entropy if needed
plt.xticks(np.arange(1, num_epochs + 1, 50.0))
plt.legend()
plt.savefig(f"two_layer_nn_train_{loss_mode}.png")
plt.show()