import cv2
import time
from PIL import Image
import numpy as np

# Load pre-trained Haar Cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load and prepare the image
def load_image(image_path):
    img = Image.open(image_path)
    return img

def resize_image(img, size):
    img = img.resize(size, Image.LANCZOS)  # Use Image.LANCZOS instead of ANTIALIAS
    return np.array(img)

# Start video capture
cap = cv2.VideoCapture(0)

# Time tracking
last_eye_detection_time = time.time()
eye_detection_timeout = 5  # seconds

# Load the image
image_path = 'image.png'
overlay_image = load_image(image_path)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    eyes_detected = False

    for (x, y, w, h) in faces:
        # Region of interest for eyes
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) > 0:
            eyes_detected = True
            last_eye_detection_time = time.time()
        
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    if not eyes_detected and (time.time() - last_eye_detection_time) > eye_detection_timeout:
        # Resize the image to match the frame size
        frame_height, frame_width = frame.shape[:2]
        resized_image = resize_image(overlay_image, (frame_width, frame_height))
        
        # Convert the image to BGR format for OpenCV
        frame[:] = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
    
    # Display the resulting frame
    cv2.imshow('Eye Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
