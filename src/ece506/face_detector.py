import cv2
from cv2 import FaceDetectorYN
from pathlib import Path

model = Path('./src/models/face_detection_yunet_2023mar.onnx')

cv2.FaceDetectorYN.create(str(model),
                          "", 
                          (300, 300),
                          score_threshold=0.5)


print("Face detector model loaded from:", model)
