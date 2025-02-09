# # import the necessary packages
from config import NMS_THRESH, MIN_CONF, People_Counter
import numpy as np
import cv2
import csv
import time

def detect_people(frame, net, ln, personIdx=0):
    # Grab the dimensions of the frame and initialize the list of results
    (H, W) = frame.shape[:2]
    results = []

    # Construct a blob from the input frame and perform a forward pass of the YOLO object detector
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # Initialize lists of detected bounding boxes, centroids, and confidences
    boxes = []
    centroids = []
    confidences = []

    # Loop over each of the layer outputs
    for output in layerOutputs:
        # Loop over each of the detections
        for detection in output:
            # Extract the class ID and confidence of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filter detections by ensuring that the object detected was a person
            if classID == personIdx and confidence > MIN_CONF:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update our list of bounding box coordinates, centroids, and confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    # Define zone thresholds
    red_zone_threshold = H // 3          # End of red zone
    orange_zone_threshold = 2 * (H // 3) # End of orange zone

    # Compute the total people counter
    if People_Counter:
        human_count = "Human count: {}".format(len(idxs))
        cv2.putText(frame, human_count, (470, frame.shape[0] - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 2)

        # Check timestamp and write to CSV
        t = time.strftime("%Y-%m-%d %H:%M:%S")
        row = [t, len(idxs)]

        # Open the CSV file in append mode and write the new row
        with open("people1_count.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        with open("present_count.csv", "w+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'People Count'])
            writer.writerow(row)

    # Ensure at least one detection exists
    if len(idxs) > 0:
        # Loop over the indexes we are keeping
        for i in idxs.flatten():
            # Extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            (cX, cY) = centroids[i]

            # Determine zone based on y-coordinate
            if cY < red_zone_threshold:
                zone_color = "Red"
                color = (0, 0, 255)  # Red color
            elif cY < orange_zone_threshold:
                zone_color = "Orange"
                color = (0, 165, 255)  # Orange color
            else:
                zone_color = "Green"
                color = (0, 255, 0)  # Green color

            # Draw the bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, zone_color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update our results list to consist of the person prediction probability, bounding box coordinates, and the centroid
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    # Return the list of results
    return results