from mylib import config, thread
from mylib.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os
import argparse
import sys
import requests
from imutils.video import FPS
from twilio.rest import Client

# Twilio configuration and SMS alert function
TWILIO_SID = "Cacfb5efla315b9a0c2554fa475718d5da"
TWILIO_AUTH_TOKEN = "fhildbabb0410@lech31089adb4d10c"
TWILIO_PHONE_NUMBER = "+17755725675"
DEST_PHONE_NUMBER = "+919989938073"

def send_sms_alert(human_count, latitude=None, longitude=None):
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    
    # Create the message body
    if latitude is not None and longitude is not None:
        message_body = (f"Alert: {human_count} crowd detected at coordinates "
                        f"Latitude: {latitude}, Longitude: {longitude}!")
    else:
        message_body = f"Alert: {human_count} crowd detected!"

    message = client.messages.create(
        body=message_body,
        from_=TWILIO_PHONE_NUMBER,
        to=DEST_PHONE_NUMBER
    )
    print(f"[INFO] SMS sent: {message.sid}")

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


def detect(video_source, output="", display=1, is_camera=False):
    # Define the base path to the YOLO directory
    base_path = os.path.abspath(os.path.join("crowd_detection1", "yolo"))

    # Load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.join(base_path, "coco.names")
    LABELS = open(labelsPath).read().strip().split("\n")

    # Derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.join(base_path, "yolov3.weights")
    configPath = os.path.join(base_path, "yolov3.cfg")

    # Load our YOLO object detector trained on COCO dataset (80 classes)
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # Check if we are going to use GPU
    if config.USE_GPU:
        print("[INFO] Looking for GPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Determine only the output layer names that we need from YOLO
    ln = net.getLayerNames()
    unconnected_layers = net.getUnconnectedOutLayers()
    ln = [ln[i[0] - 1] if isinstance(unconnected_layers[0], np.ndarray) else ln[i - 1] for i in unconnected_layers]

    # Open the video source
    if is_camera:
        vs = cv2.VideoCapture(0)
    else:
        vs = cv2.VideoCapture(video_source)
    
    if not vs.isOpened():
        print(f"Cannot open video source {video_source}")
        sys.exit(1)

    writer = None
    fps = FPS().start()

    serious = set()  # Initialize the set for serious violations

    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            print("No more frames to read.")
            break

        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

        abnormal = set()

        if len(results) >= 2:
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    if D[i, j] < config.MIN_DISTANCE:
                        serious.add(i)
                        serious.add(j)
                    if (D[i, j] < config.MAX_DISTANCE) and not serious:
                        abnormal.add(i)
                        abnormal.add(j)

        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)
            if i in serious:
                color = (0, 0, 255)
            elif i in abnormal:
                color = (0, 255, 255)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 2)

        # Display the number of serious and abnormal violations
        text = "Total serious violations: {}".format(len(serious))
        cv2.putText(frame, text, (10, frame.shape[0] - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

        text1 = "Total abnormal violations: {}".format(len(abnormal))
        cv2.putText(frame, text1, (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)

        # Check if serious violations exceed 20 and send SMS alert
        if len(serious) > 15:  # More than 20 serious violations (people in red)
            print("[INFO] Serious violations exceed 20! Sending SMS alert.")
            
            if is_camera:  # Webcam input - send location info
                # Get current location and send SMS
                latitude, longitude = get_current_location()
                if latitude is not None and longitude is not None:
                    send_sms_alert(len(serious), latitude, longitude)
            else:  # Video input - don't send location info
                send_sms_alert(len(serious))

            # Optionally, reset the serious violations counter after sending the message
            serious.clear()  # Clear the serious violations set if you want to track fresh violations

        if len(serious) >= config.Threshold:
            cv2.putText(frame, "-ALERT: Violations over limit-", (10, frame.shape[0] - 80),
                        cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 2)
        if config.ALERT:
            print("Sending mail...")  # Placeholder for mail functionality
            print('Mail sent')
            # Reset alert flag after sending email or alert
            config.ALERT = False

        if display > 0:
            cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        fps.update()

        if output != "" and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output, fourcc, 25,
                                     (frame.shape[1], frame.shape[0]), True)

        if writer is not None:
            writer.write(frame)

    fps.stop()
    print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
        is_camera = video_source.lower() == "webcam"
    else:
        video_source = "webcam"
        is_camera = True

    detect(video_source, output="", display=1, is_camera=is_camera)