from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import face_recognition

app = Flask(__name__)
socketio = SocketIO(app)

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

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('response', {'data': 'Connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('request_prediction')
def process_frames():
    while True:
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

            # Emit the processed frame and name to the connected clients
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            socketio.emit('prediction', {'image': frame_bytes, 'name': name, 'confidence': 100})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)
