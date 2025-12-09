# import cv2
# import face_detector as fd


# def crop_face(image, x, y, w, h):
#     cropped_face = image[(y):(y+h), (x):(x+w)]
#     large_dim = max(w, h)
#     cropped_face = cv2.resize(cropped_face, (large_dim, large_dim))
#     return cropped_face

# image = fd.read_image('./Figures/elon.png')

# faces = fd.face_detector.detect(image)

# fd.visualize(image, faces)

# # Save results if save is true
# print('Results saved to result.jpg\n')
# cv2.imwrite('detection_result.jpg', image)
 
# # Visualize results in a new window
# cv2.imshow("image1", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# if __name__ == "__main__":
#     print("Face cropping module loaded successfully.")

import cv2
import face_detector as fd   # your module

def crop_face(image, x, y, w, h):
    # Basic crop
    cropped = image[y:y+h, x:x+w]

    # Resize to square using the largest dimension
    L = max(w, h)
    cropped = cv2.resize(cropped, (L, L))
    return cropped


# --- Load Image ---
image = fd.read_image('./Figures/elon.png')

# --- Detect Faces ---
faces = fd.face_detector.detect(image)

# faces[1] contains an array of detections
detections = faces[1]

if detections is not None:
    for i, det in enumerate(detections):
        x, y, w, h = det[:4].astype(int)  # extract bounding box

        # Crop the face
        cropped = crop_face(image, x, y, w, h)

        # Save cropped face
        cv2.imwrite(f"cropped_face_{i}.jpg", cropped)

        # Show cropped face in a window
        cv2.imshow(f"Cropped Face {i}", cropped)

# Draw detections on original image
fd.visualize(image, faces)

cv2.imshow("Original with Boxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Done. Cropped images saved.")
