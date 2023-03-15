"""
The upload user module provides functionality for onboarding
new users to the facial recognition system (uses OpenCV)  

Author: J-A-Collins
"""

# Imports
import cv2
import math
from utils import write_on_frame
import face_detect


def capture_onboarding_image(countdown_time=5, file_name="onboarding_image.png"):
    """
    Capture an onboarding image using the default camera.
    
    Parameters
    ----------
        countdown_time:
            Countdown time in seconds before capturing the image (default: 5)

        file_name:
            Name of the file to save the captured image (default: 'onboarding_image.png')
    
    Returns
    -------
        None

    """

    video_capture = cv2.VideoCapture(0)
    try:
        while True:
            _, frame = video_capture.read()
            frame, face_box, face_coords = detect_faces(frame)
            text = f"Image will be taken in {math.ceil(countdown_time)}.."
            if face_box is not None:
                frame = write_on_frame(frame, text, face_coords[0], face_coords[1] - 10)
            cv2.imshow("Video", frame)
            cv2.waitKey(1)
            countdown_time -= 0.1
            if countdown_time <= 0:
                cv2.imwrite(file_name, face_box)
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
    print("User upload image complete.")
