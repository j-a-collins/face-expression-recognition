"""
A facial detection module that uses OpenCV's Haar Cascade Classifier
for image analysis.

Author: J-A-Collins
"""

# Imports
import cv2
import os

face_cascade = cv2.CascadeClassifier("haarcascade_fface_def.xml")


def detect_faces(img, draw_box=True):
    """
    Detect faces in an image using OpenCV's Haar Cascade Classifier.
    
    Parameters
    ----------
        img:
            Input image (BGR format)

        draw_box:
            Boolean flag to draw bounding box around detected faces

    Returns
    -------
        A tuple containing the original image with bounding boxes (if draw_box is True),
        the cropped face image, and the face coordinates [x, y, w, h]
        
    """
    # Convert image to greyscale
    greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        greyscale_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    face_box, face_coords = None, []

    for (x, y, w, h) in faces:
        if draw_box:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
        face_box = img[y : y + h, x : x + w]
        face_coords = [x, y, w, h]

    return img, face_box, face_coords


if __name__ == "__main__":
    files = os.listdir("sample_faces")
    images = [file for file in files if "jpg" in file]

    for image in images:
        img = cv2.imread(f"sample_faces/{image}")
        detected_faces, _, _ = detect_faces(img)
        cv2.imwrite(f"sample_faces/detected_faces/{image}", detected_faces)
