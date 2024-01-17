import cv2
import face_recognition

def get_face_encodings(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    return face_locations, face_encodings

# Load known faces and labels
known_faces = []
known_labels = ["orang", "ojan", "obam", "arya"]

for i in range(1, 5):
    known_image_path = f"{i}.jpg"  # Replace with the correct path
    known_image = cv2.imread(known_image_path)
    _, known_encoding = get_face_encodings(known_image)
    known_faces.append(known_encoding[0])

# Open a connection to the webcam (you may need to change the index based on your system)
video_capture = cv2.VideoCapture(1)

while True:
    # Capture each frame from the webcam
    ret, frame = video_capture.read()

    # Find face locations and face encodings in the current frame
    unknown_locations, unknown_encodings = get_face_encodings(frame)

    # Iterate through each unknown face and check if it matches any known face
    for unknown_face_location, unknown_face_encoding in zip(unknown_locations, unknown_encodings):
        matches = face_recognition.compare_faces(known_faces, unknown_face_encoding)
        name = "Unknown"

        for i, match in enumerate(matches):
            if match:
                name = known_labels[i]
                break

        # Draw rectangle around the face and display the name
        top, right, bottom, left = unknown_face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Webcam Face Recognition', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
