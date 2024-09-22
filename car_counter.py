import numpy as np
from ultralytics import YOLO
import cv2
import math

# Open video file or camera feed
cap = cv2.VideoCapture(r"C:\Users\Aung-Myat\Videos\object detection\cars.mp4")

# Set video frame size to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load YOLOv8 model
model = YOLO("yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Vehicle tracking dictionary to keep track of car positions and their last known positions
vehicles = {}
vehicle_id = 0

# Define the region where vehicles cross
limits = [200, 400, 440, 400]
totalCount = 0
min_movement_threshold = 10  # Minimum movement distance to be considered "moving"


# Function to detect if a vehicle crosses the line
def is_vehicle_crossing_line(vehicle_position, line_limits):
    cx, cy = vehicle_position
    if line_limits[0] < cx < line_limits[2] and line_limits[1] - 15 < cy < line_limits[1] + 15:
        return True
    return False


while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)

    current_frame_vehicles = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                # Get the center point of the bounding box (to track the vehicle's movement)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                current_frame_vehicles.append((cx, cy))

    # Track vehicles across frames and check for movement
    for (cx, cy) in current_frame_vehicles:
        vehicle_found = False
        for vehicle_id, (prev_cx, prev_cy) in vehicles.items():
            # Calculate distance between previous position and current position
            distance = math.hypot(cx - prev_cx, cy - prev_cy)
            if distance < min_movement_threshold:
                # Update vehicle's position if it's moving and not too far from its previous position
                vehicles[vehicle_id] = (cx, cy)
                vehicle_found = True
                # Check if the vehicle crosses the line
                if is_vehicle_crossing_line((cx, cy), limits):
                    if vehicle_id not in vehicles:
                        totalCount += 1
                        print(f"Vehicle {vehicle_id} counted. Total count: {totalCount}")
                break

        # If no matching vehicle was found, assign a new ID to this vehicle
        if not vehicle_found:
            vehicle_id += 1
            vehicles[vehicle_id] = (cx, cy)

    # Remove vehicles that have exited the frame
    vehicles = {vid: pos for vid, pos in vehicles.items() if pos[1] < img.shape[0]}

    # Draw line and display the vehicle count
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    cv2.putText(img, f'Count: {totalCount}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    # Draw the detected vehicle bounding boxes and ids
    for vehicle_id, (cx, cy) in vehicles.items():
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        cv2.putText(img, f'{vehicle_id}', (cx, cy - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
