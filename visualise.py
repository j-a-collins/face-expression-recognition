"""
This module loads and visualises images of faces from the "faces" dataset, 
which contains images of different users. The dataset is divided into training 
and test sets, with the first 35 users used for training and the remaining 
subjects used for testing.

Author: J-A-Collins
"""


# Imports
import matplotlib

matplotlib.use("TkAgg")
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.image import load_img, img_to_array


def load_data(faces_dir):
    """
    Load image data and corresponding labels from the specified directory.
    
    Parameters
    ----------

        faces_dir: str
            The path to the directory containing the image data.
        
    Returns
    -------

        tuple:
            Four numpy arrays containing the training and test images and their corresponding labels.

    """
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    subfolders = sorted([file.path for file in os.scandir(faces_dir) if file.is_dir()])
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
    return X_train, Y_train, X_test, Y_test


def plot_subject_images(X_train, Y_train, subject_idx):
    """
    Plot images of a specific subject from the training dataset.
    
    Parameters
    ----------

        X_train: numpy.ndarray
            The array of training images.

        Y_train: numpy.ndarray
            The array of training labels.

        subject_idx: int
            The index of the subject whose images to plot.

    """
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(
        3, 3, figsize=(10, 10)
    )
    subject_img_idx = np.where(Y_train == subject_idx)[0].tolist()
    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]):
        img = X_train[subject_img_idx[i]]
        img = img.reshape(img.shape[0], img.shape[1])
        ax.imshow(img, cmap="gray")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_first_images(X_train, Y_train, subjects):
    fig, axes = plt.subplots(3, 3, figsize=(10, 12))
    subject_img_idx = [np.where(Y_train == i)[0].tolist()[0] for i in subjects]

    for i, ax in enumerate(axes.flat):
        img = X_train[subject_img_idx[i]]
        img = img.reshape(img.shape[0], img.shape[1])
        ax.imshow(img, cmap="gray")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Subject {i}")

    plt.tight_layout()
    plt.show()


faces_dir = "faces/"
X_train, Y_train, X_test, Y_test = load_images(faces_dir)

subject_idx = 4
plot_subject_images(X_train, Y_train, subject_idx)

subjects = range(10)
plot_first_images(X_train, Y_train, subjects)
