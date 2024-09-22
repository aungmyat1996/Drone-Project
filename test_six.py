import cv2
from ultralytics import YOLO
import time

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize video capture
cap = cv2.VideoCapture('track_car.mp4')  # Use 0 for webcam

# Initialize tracker
tracker = cv2.TrackerKCF_create()
tracking = False
track_box = None

# Placeholder values for the new parameters
wind_speed = 0  # Assuming no wind for testing
wind_direction = 'N/A'  # Placeholder if no data available
object_speed = 0  # Assuming stationary object for testing
altitude = 0  # Placeholder if no altitude data available

# Initialize variables for FPS calculation and flight time
start_time = time.time()
frame_count = 0
fps = 0

def calculate_fps():
    """
    Calculates the Frames Per Second (FPS).
    """
    global frame_count, start_time, fps
    frame_count += 1
    if frame_count >= 10:  # Update every 10 frames
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        start_time = end_time
        frame_count = 0
    return fps

def draw_target_display(frame, x, y, w, h, color=(0, 255, 0), thickness=2):
    """
    Draws corner brackets and crosshair around the detected object.
    """
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

def draw_hud(frame, battery, mode, fps, flight_time, altitude, wind_speed, wind_direction, object_speed):
    font_scale = 0.5  # Smaller font scale for HUD elements
    # Existing elements
    cv2.putText(frame, f'Battery: {battery}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
    cv2.putText(frame, f'Mode: {mode}', (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
    cv2.putText(frame, f'Flight Time: {flight_time:.1f}s', (frame.shape[1] - 250, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
    # New elements
    cv2.putText(frame, f'Altitude: {altitude}m', (frame.shape[1] - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
    cv2.putText(frame, f'Object Speed: {object_speed} km/h', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not tracking:
        # Perform YOLOv8 detection
        results = model(frame)

        # Process the detection results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0].item())  # Get the class ID
                if model.names[cls] == 'car':  # Check if the detected object is a car
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates
                    track_box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

                    # Initialize tracker with the first car detected
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
            tracking = False  # Reset tracking if it fails

    # Calculate the current FPS
    current_fps = calculate_fps()

    # Simulated telemetry data (replace with actual data fetching from drone)
    battery = 85  # Placeholder for battery percentage
    mode = "GPS"  # Placeholder for drone mode
    flight_time = time.time() - start_time  # Calculate flight time
    attitude = (5.5, -3.2, 0.8)  # Placeholder for pitch, roll, yaw values

    # Updated function call with all required parameters
    draw_hud(frame, battery, mode, current_fps, flight_time, altitude, wind_speed, wind_direction, object_speed)

    # Display the frame
    cv2.imshow('YOLOv8 Car Tracking with Drone HUD', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


