import cv2
import numpy as np
import face_recognition
import torch
import time
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from pygame import mixer
from pymongo import MongoClient
from datetime import datetime
from ultralytics import YOLO

# Ensure directories exist for storing photos and uploads
if not os.path.exists("captured_images"):
    os.makedirs("captured_images")
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# MongoDB Setup
client = MongoClient("mongodb://localhost:27017/")  # Connect to MongoDB server
db = client["intruder_animal_detection"]  # Database name
alerts_collection = db["alerts"]  # Collection for storing alerts

# Telegram Bot Configuration
BOT_TOKEN = "7726687998:AAHHkM8x8bwo3PDsw7Of1jCHeSZGBORn0q8"
CHAT_ID = "812723861"

# Alarm sound paths
DOG_ALARM_PATH = "sounds/dog_alarm.mp3"
CAT_ALARM_PATH = "sounds/cat_alarm.mp3"
ELE_ALARM_PATH = "sounds/elephant_alarm.mp3"

# Initialize pygame mixer
mixer.init()

# Load YOLOv8 model for animal detection
animal_model = YOLO("yolov8n.pt")

# Load YOLOv5 model for intruder detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
intruder_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True).to(device)
intruder_model.eval()
torch.set_grad_enabled(False)

# Load known faces
known_face_encodings = []
known_face_names = ["Family Member 1", "Family Member 2", "Family Member 3", "Family Member 4"]
family_images = ["family_member_1.jpg", "family_member_2.jpg", "jk.jpg", "saleel.jpg"]

for i, file_name in enumerate(family_images):
    image_path = f"family_faces/{file_name}"
    if not os.path.exists(image_path):
        print(f"Error: File not found - {image_path}")
        continue

    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    
    if len(encodings) > 0:
        known_face_encodings.append(encodings[0])  # Take first encoding
        print(f"Loaded face encoding for {known_face_names[i]}")
    else:
        print(f"Warning: No face found in {file_name}")

print(f"Total loaded known faces: {len(known_face_encodings)}")

# Set up webcam feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)  # Set higher FPS for smoother video

frame_counter = 0

# Cooldown variables
alert_cooldown = 10  # Seconds before a new alert can be triggered
last_alert_time = 0

executor = ThreadPoolExecutor(max_workers=2)  # Multi-threading

# Function to send an alert via Telegram
def send_telegram_alert(image_path, chat_id, token, message):
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(image_path, "rb") as photo:
        payload = {"chat_id": chat_id, "caption": message}
        files = {"photo": photo}
        response = requests.post(url, data=payload, files=files)
        return response.json()

# Function to send alert to MongoDB and control server
def send_alert_to_server(alert_message, image_path, alarm_type=None):
    # Store alert in MongoDB
    alert_data = {
        "alert_message": alert_message,
        "image_path": image_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Add timestamp
        "alarm_type": alarm_type
    }
    try:
        alerts_collection.insert_one(alert_data)  # Insert alert data into MongoDB
        print("Alert stored in MongoDB!")
    except Exception as e:
        print(f"Error storing alert in MongoDB: {e}")

# Function to play alarm sound
def play_alarm(sound_file):
    try:
        mixer.music.load(sound_file)
        mixer.music.play(loops=-1)  # Loop the alarm until stopped
    except Exception as e:
        print(f"Error playing sound: {e}")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    frame_resized = cv2.resize(frame, (640, 480))
    small_frame = cv2.resize(frame, (320, 240))  # Lower resolution for face recognition
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Run YOLO for intruder detection
    future_intruder = executor.submit(intruder_model, frame_resized)
    future_faces = executor.submit(face_recognition.face_encodings, rgb_frame, face_recognition.face_locations(rgb_frame))

    # Run YOLOv8 for animal detection
    animal_results = animal_model.predict(frame_resized, conf=0.5)

    # Process intruder detection
    intruder_results = future_intruder.result()
    boxes = intruder_results.xyxy[0].cpu().numpy()
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        if conf > 0.4:  # Lower confidence threshold to avoid missing detections
            label = intruder_model.names[int(cls)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # If a person is detected, send an alert
            if label == "person":
                current_time = time.time()
                if current_time - last_alert_time > alert_cooldown:
                    alert_message = "ALERT: Person detected!"
                    last_alert_time = current_time
                    filename = f"captured_images/person_{int(current_time)}.jpg"
                    cv2.imwrite(filename, frame)
                    response = send_telegram_alert(filename, CHAT_ID, BOT_TOKEN, alert_message)
                    send_alert_to_server(alert_message, filename, "intruder")
                    print("Telegram Response:", response)

    # Process face recognition
    face_encodings = future_faces.result()
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            print(f"Matched: {name}")
        else:
            print("No match found.")

        # Alert if unknown
        if name == "Unknown":
            current_time = time.time()
            if current_time - last_alert_time > alert_cooldown:
                alert_message = "ALERT: Unknown person detected!"
                last_alert_time = current_time
                filename = f"captured_images/unknown_{int(current_time)}.jpg"
                cv2.imwrite(filename, frame)
                response = send_telegram_alert(filename, CHAT_ID, BOT_TOKEN, alert_message)
                send_alert_to_server(alert_message, filename, "unknown_intruder")
                print("Telegram Response:", response)
            color = (0, 0, 255)  # Red for unknown
        else:
            color = (255, 0, 0)  # Blue for recognized

        # Draw rectangle & name
        (top, right, bottom, left) = face_recognition.face_locations(rgb_frame)[0]
        cv2.rectangle(frame, (left * 2, top * 2), (right * 2, bottom * 2), color, 2)
        cv2.putText(frame, name, (left * 2, top * 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Process animal detection
    for result in animal_results[0].boxes:
        cls = int(result.cls[0])
        conf = float(result.conf[0])
        label = animal_results[0].names[cls]
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Actions for specific animals
        if label.lower() == "dog" and conf > 0.5:
            alert_message = "⚠️ Warning: Dog detected!"
            filename = f"captured_images/dog_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            send_telegram_alert(filename, CHAT_ID, BOT_TOKEN, alert_message)
            send_alert_to_server(alert_message, filename, "dog")
            play_alarm(DOG_ALARM_PATH)
        elif label.lower() == "cat" and conf > 0.3:
            alert_message = "🚨 Alert: Cat detected!"
            filename = f"captured_images/cat_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            send_telegram_alert(filename, CHAT_ID, BOT_TOKEN, alert_message)
            send_alert_to_server(alert_message, filename, "cat")
            play_alarm(CAT_ALARM_PATH)
        elif label.lower() == "elephant" and conf > 0.3:
            alert_message = "🚨 Alert: Elephant detected!"
            filename = f"captured_images/elephant_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            send_telegram_alert(filename, CHAT_ID, BOT_TOKEN, alert_message)
            send_alert_to_server(alert_message, filename, "elephant")
            play_alarm(ELE_ALARM_PATH)

    # Display output
    cv2.imshow("Intruder and Animal Detection System", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()