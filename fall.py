import cv2
import cvzone
import math
from ultralytics import YOLO
import sys
import requests
from twilio.rest import Client

# Twilio configuration and SMS alert function
TWILIO_SID = "Cacfb5efla315b9a0c2554fa475718d5da"
TWILIO_AUTH_TOKEN = "fhildbabb0410@lech31089adb4d10c"
TWILIO_PHONE_NUMBER = "+17755725675"
DEST_PHONE_NUMBER = "+919989938073"

def send_alert_sms(message_body):
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body=message_body,
        from_=TWILIO_PHONE_NUMBER,
        to=DEST_PHONE_NUMBER
    )
    print(f'SMS sent: {message.sid}')

# Function to get current location for alert
def get_current_location():
    try:
        response = requests.get('http://ipinfo.io/json')
        data = response.json()
        location = data['loc'].split(',')
        latitude = float(location[0])
        longitude = float(location[1])
        return latitude, longitude
    except Exception as e:
        print(f"[ERROR] Unable to get current location: {e}")
        return None, None

# Function to classify posture (standing/sitting) based on bounding box dimensions
def is_person_sitting(height, width):
    aspect_ratio = height / width
    return aspect_ratio < 1.2  # Adjust this threshold based on your observations

# Frame processing function for fall detection
def process_frame(frame, person_model, fall_model, classnames_person, classnames_fall, is_webcam):
    # Detect persons with YOLOv8n model
    person_results = person_model(frame)
    fall_detected = False
    latitude, longitude = None, None

    if is_webcam:
        latitude, longitude = get_current_location()

    # Process the results for "person" detection
    for result in person_results:
        parameters = result.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect_name = classnames_person[class_detect]
            conf = math.ceil(confidence * 100)

            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            # Display bounding box for detected "person"
            if class_detect_name == 'person' and conf > 80:
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect_name} {conf}%', [x1, y1 - 12], thickness=2, scale=2)

                if is_person_sitting(height, width):
                    #cvzone.putTextRect(frame, 'Sitting', [x1, y1 - 40], thickness=2, scale=1.5, colorR=(255, 165, 0))
                    continue  # Skip fall detection for sitting individuals

            # Now use best1.pt for fall detection
            fall_results = fall_model(frame)  # Fall detection on the entire frame
            for fall_result in fall_results:
                fall_parameters = fall_result.boxes
                for fall_box in fall_parameters:
                    fall_x1, fall_y1, fall_x2, fall_y2 = fall_box.xyxy[0]
                    fall_x1, fall_y1, fall_x2, fall_y2 = int(fall_x1), int(fall_y1), int(fall_x2), int(fall_y2)
                    fall_confidence = fall_box.conf[0]
                    fall_class_detect = fall_box.cls[0]
                    fall_class_detect = int(fall_class_detect)
                    fall_conf = math.ceil(fall_confidence * 100)

                    # Debug: Print fall detection class and confidence
                    #print(f"Fall Class: {fall_class_detect}, Confidence: {fall_conf}%")

                    # Check if fall detection is within class range
                    if fall_class_detect >= len(classnames_fall):
                        print(f"[ERROR] fall_class_detect {fall_class_detect} is out of range. Skipping this fall detection.")
                        continue

                    fall_class_name = classnames_fall[fall_class_detect]

                    # Fall detection logic based on best1.pt
                    if fall_class_name == 'fall' and fall_conf > 80:
                        cvzone.putTextRect(frame, 'Fall Detected', [fall_x1, fall_y1 - 40], thickness=2, scale=2, colorR=(0, 0, 255))
                        fall_detected = True

    # If fall detected, send alert SMS
    if fall_detected:
        message_body = f"Fall Detected! Location: Latitude {latitude}, Longitude {longitude}" if is_webcam else "Fall Detected"
        print("sms sent")
        send_alert_sms(message_body)

    return frame

# Function to start fall detection from a video file
def start_fall_detection(video_path, frame_skip=2):
    person_model = YOLO('yolov8n.pt')  # Using YOLOv8n model for person detection
    fall_model = YOLO('best1.pt')  # Your custom fall detection model

    # Load class names for YOLOv8n (person detection) and best1.pt (fall detection)
    with open('classes.txt', 'r') as f:
        classnames_person = f.read().splitlines()

    with open('classes1.txt', 'r') as f:
        classnames_fall = f.read().splitlines()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video stream or file.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (640, 480))
        frame = process_frame(frame, person_model, fall_model, classnames_person, classnames_fall, is_webcam=False)

        cv2.imshow('Fall Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to start fall detection using a webcam
def start_webcam_detection(frame_skip=2):
    person_model = YOLO('yolov8n.pt')  # Using YOLOv8n model for person detection
    fall_model = YOLO('best1.pt')  # Your custom fall detection model

    # Load class names for YOLOv8n (person detection) and best1.pt (fall detection)
    with open('classes.txt', 'r') as f:
        classnames_person = f.read().splitlines()

    with open('classes1.txt', 'r') as f:
        classnames_fall = f.read().splitlines()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (640, 480))
        frame = process_frame(frame, person_model, fall_model, classnames_person, classnames_fall, is_webcam=True)

        cv2.imshow('Fall Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == '_main_':
    if len(sys.argv) != 2:
        print("Usage: python fall.py [video_path|webcam]")
        sys.exit(1)

    input_source = sys.argv[1].strip().lower()
    if input_source == 'webcam':
        start_webcam_detection()
    else:
        start_fall_detection(input_source)