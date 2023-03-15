"""
System module for real-time face recognition using a the pre-trained 
Siamese NN. The worflow: capture video from webcam, detects a face, and 
then identify the user by comparing the detected face with a reference
image.

Author: J-A-Collins
"""


# Imports
import os
import sys
import cv2
import utils
import collections
from keras.models import load_model
from face_detect import detect_faces


# Constants and variables
IDENTITY_THRESHOLD = 0.3


def process_image(img):
    """
    Preprocess an input image for the Siamese NN.
    
    This function converts the input image to grayscale, normalises pixel
    values to the range [0, 1], resizes the image to (92, 112), and reshapes
    the image to have the proper dimensions for the neural network input.
    
    Parameters
    ----------
        img: (numpy.ndarray)
            The input image to be preprocessed.
        
    Returns
    -------
        numpy.ndarray: 
            The preprocessed image ready to be fed to the neural network.

    """
    # Check if the input image has 3 channels (BGR)
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype("float32") / 255
    img = cv2.resize(img, (92, 112))
    img = img.reshape(1, img.shape[0], img.shape[1], 1)
    return img


name = input("Enter user name: ")
print(f"Hello {name}. Face recog system is now running. Press 'q' to quit.")

# Validate model training process has completed:
files = os.listdir()
if "nn.h5" not in files:
    print("Error: Pre-trained Neural Network not found!")
    print("Please run nn.py first")
    sys.exit()

# Validate upload process has completed:
if "true_img.png" not in files:
    print("Error: True image not found.")
    print("Please run upload_user.py first")
    sys.exit()

# load pre-trained Siamese neural network
model = load_model(
    "nn.h5",
    custom_objects={
        "contrastive_loss": utils.contrastive_loss,
        "euclidean_distance": utils.euclidean_distance,
    },
)

# prepare the true image obtained during upload process
true_img = cv2.imread("true_img.png", 0)
true_img = process_image(true_img)

video_capture = cv2.VideoCapture(0)
predictions = collections.deque(maxlen=15)


while True:
    # Capture frame-by-frame
    _, frame = video_capture.read()

    # Detect Faces
    frame, face_img, face_coords = detect_faces(frame, draw_box=False)

    if face_img is not None:
        face_img = process_image(face_img)
        predictions.append(1 - model.predict([true_img, face_img])[0][0])
        x, y, w, h = face_coords
        if len(predictions) == 15 and sum(predictions) / 15 >= IDENTITY_THRESHOLD:
            text = "Identity: {}".format(name)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
        elif len(predictions) < 15:
            text = "Identifying ..."
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 5)
        else:
            text = "Identity Unknown!"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
        frame = utils.write_on_frame(frame, text, face_coords[0], face_coords[1] - 10)

    else:
        predictions = collections.deque(
            maxlen=15
        )  # clear existing preds if no detection

    # Display the resulting frame
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release capture:
video_capture.release()
cv2.destroyAllWindows()
