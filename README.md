# face-expression-recognition
A small ML program that makes use of tensorflow and OpenCV for recognition of facial expressions.

This program makes use of [Cascade Classifier Training (CCT)](https://www.researchgate.net/publication/220660094_Robust_Real-Time_Face_Detection). The code uses pretrained Haar Cascade models to detect faces in the live feed. The necessary XML file is loaded using the CascadeClassifier loading method. The detection is then performed by the detectMultiScale method, which returns boundary rectangles for the detected face. Since the Haar Cascade function is trained on images in grayscale, the code also converts each frame from the feed into grayscale - these are then cropped for the faces in the image. 
