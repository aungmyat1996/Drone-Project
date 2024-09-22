import cv2
from ultralytics import YOLO
import numpy as np

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize video capture
cap = cv2.VideoCapture('track_car.mp4')  # or 0 for webcam

# Initialize tracker
tracker = cv2.TrackerKCF_create()
tracking = False
track_box = None

def draw_target_display(frame, x, y, w, h, color=(0, 255, 0), thickness=1):
    # Draw corner brackets
    l = min(w, h) // 4  # length of the bracket
    
    # Top-left
    cv2.line(frame, (x, y), (x + l, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + l), color, thickness)
    
    # Top-right
    cv2.line(frame, (x + w, y), (x + w - l, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + l), color, thickness)
    
    # Bottom-left
    cv2.line(frame, (x, y + h), (x + l, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - l), color, thickness)
    
    # Bottom-right
    cv2.line(frame, (x + w, y + h), (x + w - l, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - l), color, thickness)
    
    # Draw crosshair
    center_x, center_y = x + w // 2, y + h // 2
    size = min(w, h) // 4
    cv2.line(frame, (center_x - size, center_y), (center_x + size, center_y), color, thickness)
    cv2.line(frame, (center_x, center_y - size), (center_x, center_y + size), color, thickness)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not tracking:
        # Perform YOLOv8 detection
        results = model(frame)
        
        # Find the first car detected
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if model.names[cls] == 'car':
                    x1, y1, x2, y2 = box.xyxy[0]
                    track_box = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, track_box)
                    tracking = True
                    break
            if tracking:
                break
    
    if tracking:
        # Update tracker
        success, track_box = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in track_box]
            draw_target_display(frame, x, y, w, h)
        else:
            tracking = False

    cv2.imshow('YOLOv8 Car Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()