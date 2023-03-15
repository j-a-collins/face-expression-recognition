"""
Helper functions for the recognition system

Author: J-A-Collins
"""
import numpy as np
import random
import os
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array


def euclidean_distance(vectors):
    """
    Compute the Euclidean distance between two vectors.
    
    Parameters
    ----------
        vectors: tuple
            A tuple containing two Keras tensors.
        
    Returns
    -------
        Keras tensor:
            The Euclidean distance between the input vectors.
    """

    vector1, vector2 = vectors
    sum_square = K.sum(K.square(vector1 - vector2), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def contrastive_loss(Y_true, D):
    """
    Compute the contrastive loss for Siamese networks.
    
    Parameters
    ----------
        Y_true: Keras tensor
            Ground truth labels.
        D: Keras tensor
            Predicted Euclidean distances between pairs.
        
    Returns
    -------
        Keras tensor:
            The contrastive loss.

    """
    margin = 1
    return K.mean(Y_true * K.square(D) + (1 - Y_true) * K.maximum((margin - D), 0))


def accuracy(y_true, y_pred):
    """
    Compute the classification accuracy for Siamese networks.
    
    Parameters
    ----------
        y_true: Keras tensor
            Ground truth labels.

        y_pred: Keras tensor
            Predicted Euclidean distances between pairs.
        
    Returns
    -------
        Keras tensor: The classification accuracy.

    """
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def create_pairs(X, Y, num_classes):
    """
    Create positive and negative pairs for training the Siamese network.
    
    Parameters
    ----------

        X: numpy.ndarray
            Array of images.

        Y: numpy.ndarray
            Array of class labels.

        num_classes: int
            Number of classes.
        
    Returns
    -------
        tuple:
            Two numpy arrays containing image pairs and their corresponding labels.

    """
    pairs, labels = [], []
    # Index of images in X and Y for each class
    class_idx = [np.where(Y == i)[0] for i in range(num_classes)]
    # The minimum number of images across all classes
    min_images = min(len(class_idx[i]) for i in range(num_classes)) - 1

    for c in range(num_classes):
        for n in range(min_images):
            # create positive pair
            img1 = X[class_idx[c][n]]
            img2 = X[class_idx[c][n + 1]]
            pairs.append((img1, img2))
            labels.append(1)

            # Negative pair
            neg_list = list(range(num_classes))
            neg_list.remove(c)
            # Select random class from negative list.
            neg_c = random.sample(neg_list, 1)[0]
            img1 = X[class_idx[c][n]]
            img2 = X[class_idx[neg_c][n]]
            pairs.append((img1, img2))
            labels.append(0)

    return np.array(pairs), np.array(labels)


def create_shared_network(input_shape):
    """
    Create a shared convolutional neural network for the Siamese network.
    
    Parameters
    ----------

        input_shape (tuple): The shape of the input images.
        
    Returns
    -------

        keras.models.Sequential:
            The shared convolutional neural network.

    """
    model = Sequential(name="Shared_Conv_Network")
    model.add(
        Conv2D(
            filters=64, kernel_size=(3, 3), activation="relu", input_shape=input_shape
        )
    )
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(units=128, activation="sigmoid"))
    return model


def get_data(dir):
    """
    Load the image data and corresponding labels from the specified directory.
    
    Parameters
    ----------

        dir (str): The path to the directory containing the image data.
        
    Returns
    -------

        tuple:
            Four numpy arrays containing the training and test images and their corresponding labels.
    
    """
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    subfolders = sorted([file.path for file in os.scandir(dir) if file.is_dir()])
    for idx, folder in enumerate(subfolders):
        for file in sorted(os.listdir(folder)):
            img_path = os.path.join(folder, file)
            img = load_img(img_path, color_mode="grayscale")
            img = img_to_array(img).astype("float32") / 255
            img = img.reshape(img.shape[0], img.shape[1], 1)
            if idx < 35:
                X_train.append(img)
                Y_train.append(idx)
            else:
                X_test.append(img)
                Y_test.append(idx - 35)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    return (X_train, Y_train), (X_test, Y_test)


def write_on_frame(frame, text, text_x, text_y):
    """
    Draw a text box with specified text on the input frame.
    
    Parameters
    ----------

        frame: (numpy.ndarray)
            The input frame to draw the text box on.

        text: str
            The text to be displayed in the box.

        text_x: int
            The x-coordinate of the text box's top-left corner.

        text_y: int
            The y-coordinate of the text box's top-left corner.
        
    Returns
    -------

        numpy.ndarray:
            The frame with the text box drawn on it.

    """
    (text_width, text_height) = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2
    )[0]
    box_coords = (
        (text_x, text_y),
        (text_x + text_width + 20, text_y - text_height - 20),
    )
    cv2.rectangle(frame, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
    cv2.putText(
        frame,
        text,
        (text_x, text_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 0),
        thickness=2,
    )
    return frame
