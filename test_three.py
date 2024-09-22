import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # or use a custom trained model

# Open video file
video_path = 'track_car.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create VideoWriter object
# output_path = 'output.mp4'
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
# 
# Define smaller fixed green box coordinates
box_width = int(width * 0.3)  # 30% of frame width
box_height = int(height * 0.3)  # 30% of frame height
x_offset = (width - box_width) // 2
y_offset = (height - box_height) // 2
green_box = np.array([
    [x_offset, y_offset],
    [x_offset + box_width, y_offset],
    [x_offset + box_width, y_offset + box_height],
    [x_offset, y_offset + box_height]
])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Process detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Check if the detected object is a car
            if model.names[int(box.cls)] == 'car':
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate center and radius for circular bounding box
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                radius = int(max(x2 - x1, y2 - y1) / 2)
                
                # Draw circular bounding box
                cv2.circle(frame, center, radius, (0, 255, 0), 2)

    # Draw fixed green box
    #cv2.polylines(frame, [green_box], True, (0, 255, 0), 2)

    # Add corner markers
    corner_length = 20
    cv2.line(frame, (0, 0), (corner_length, 0), (0, 255, 0), 2)
    cv2.line(frame, (0, 0), (0, corner_length), (0, 255, 0), 2)
    cv2.line(frame, (width, 0), (width - corner_length, 0), (0, 255, 0), 2)
    cv2.line(frame, (width, 0), (width, corner_length), (0, 255, 0), 2)
    cv2.line(frame, (0, height), (corner_length, height), (0, 255, 0), 2)
    cv2.line(frame, (0, height), (0, height - corner_length), (0, 255, 0), 2)
    cv2.line(frame, (width, height), (width - corner_length, height), (0, 255, 0), 2)
    cv2.line(frame, (width, height), (width, height - corner_length), (0, 255, 0), 2)

    # Write frame to output video
#     out.write(frame)

    # Display the frame (optional, for debugging)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()