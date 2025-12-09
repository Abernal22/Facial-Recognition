import cv2
from cv2 import FaceDetectorYN
from pathlib import Path
import numpy as np

model = Path('./src/models/face_detection_yunet_2023mar.onnx')

face_detector = cv2.FaceDetectorYN.create(str(model),
                          "", 
                          (300, 300),
                          score_threshold=0.5)

print("Face detector model loaded from:", model)

def visualize(input, faces, thickness=2, fps=0):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))
 
            coords = face[:-1].astype(np.int32)
            cv2.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv2.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv2.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv2.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv2.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv2.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv2.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def read_image(image_path):
    image = cv2.imread(str(image_path))
    resized_image = cv2.resize(image, (300, 300))
    return resized_image

# # Now we need to set up an image for detection
# image  = cv2.imread('./Figures/elon.png')

# resized_image = cv2.resize(image, (300, 300))

# # Run the detector on the image
# faces = face_detector.detect(resized_image)

# visualize(resized_image, faces)

# # Save results if save is true
# print('Results saved to result.jpg\n')
# cv2.imwrite('detection_result.jpg', resized_image)
 
# # Visualize results in a new window
# cv2.imshow("image1", resized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Face detector module loaded successfully.")
