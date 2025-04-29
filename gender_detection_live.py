import cv2
import numpy as np
import hashlib
import time

# Paths to model files
FACE_PROTO = "weights/deploy.prototxt"
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
GENDER_PROTO = "weights/deploy_gender.prototxt"
GENDER_MODEL = "weights/gender_net.caffemodel"

def hash_face_image(face_img):
    resized_face = cv2.resize(face_img, (100, 100))  # Normalize size
    face_bytes = resized_face.tobytes()
    return hashlib.md5(face_bytes).hexdigest()

# Model parameters
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']

# Load models
face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

# Face memory for tracking (Optional)
face_memory = {}  # face_id: (gender, last_seen_time)
expiry_duration = 10  # seconds to remember each face

def get_faces(frame, confidence_threshold=0.5):
    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    face_net.setInput(blob)
    output = np.squeeze(face_net.forward())
    faces = []
    
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            start_x, start_y, end_x, end_y = box.astype(int)
            start_x = max(0, start_x - 10)
            start_y = max(0, start_y - 10)
            end_x = min(frame.shape[1], end_x + 10)
            end_y = min(frame.shape[0], end_y + 10)
            faces.append((start_x, start_y, end_x, end_y))
    return faces

def get_gender_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    gender_net.setInput(blob)
    return gender_net.forward()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    current_time = time.time()

    # Clean up old faces from memory (optional)
    face_memory = {
        fid: (gender, t) for fid, (gender, t) in face_memory.items()
        if current_time - t < expiry_duration
    }

    # Detect faces
    faces = get_faces(frame)

    for (start_x, start_y, end_x, end_y) in faces:
        face_img = frame[start_y:end_y, start_x:end_x]
        if face_img.size == 0:
            continue

        face_id = hash_face_image(face_img)

        # Only process if this is a new face or recently expired
        if face_id not in face_memory:
            gender_preds = get_gender_predictions(face_img)
            gender = GENDER_LIST[np.argmax(gender_preds)]

            # Store face with gender and current timestamp
            face_memory[face_id] = (gender, current_time)

            # Draw rectangle and label
            label = f"{gender} ({gender_preds[0][np.argmax(gender_preds)]:.2f})"
            rect_color = (255, 0, 0) if gender == 'Male' else (255, 105, 180)  # Blue for male, pink for female
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), rect_color, 2)
            cv2.putText(frame, label, (start_x, start_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, rect_color, 2)

    # Show frame
    cv2.imshow("Gender Detection", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

