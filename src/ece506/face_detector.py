import cv2

cv2.FaceDetectorYN_create('./face_detection_yunet_2023mar.onnx',
                          "", 
                          (300, 300),
                          score_threshold=0.5)