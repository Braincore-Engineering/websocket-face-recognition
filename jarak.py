import cv2
import math

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def calculate_distance(face_width, focal_length, pixel_width):
    return (face_width * focal_length) / pixel_width

camera_index = 1

# Set the desired resolution for portrait mode
portrait_resolution = (480, 640)  # (height, width)

cap = cv2.VideoCapture(camera_index)

while True:
    ret, frame = cap.read()
    # print(frame.shape)
    # Resize the frame to portrait resolution
    frame = cv2.resize(frame, (640,640))
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        distance = calculate_distance(150, 50, w)
        cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow("Face Detection", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
