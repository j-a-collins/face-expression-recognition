"""
The upload user module provides functionality for onboarding
new users to the facial recognition system (uses OpenCV)  

Author: J-A-Collins
"""

# Imports
import cv2
import math
from utils import write_on_frame
from face_detect import detect_faces


video_capture = cv2.VideoCapture(0)
counter = 5

while True:
    _, frame = video_capture.read()
    frame, face_box, face_coords = detect_faces(frame)
    text = 'Image will be taken in {}..'.format(math.ceil(counter))
    if face_box is not None:
        frame = write_on_frame(frame, text, face_coords[0], face_coords[1]-10)
    cv2.imshow('Video', frame)
    cv2.waitKey(1)
    counter -= 0.1
    if counter <= 0:
        cv2.imwrite('true_img.png', face_box)
        break


video_capture.release()
cv2.destroyAllWindows()
print("User upload image complete.")
