import cv2
import numpy as np
import time
from ultralytics import YOLO

def draw_hud(img, fps, start_time):
    """Draws a camera viewfinder overlay with recording info and framing."""
    height, width = img.shape[:2]
    # Display REC status
    cv2.putText(img, 'REC', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Simulate battery level
    battery_level = 100  # Simulated full battery
    cv2.putText(img, f'Battery: {battery_level}%', (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display video resolution
    cv2.putText(img, '4K UHD', (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw crosshair in the center
    center_x, center_y = width // 2, height // 2
    cv2.line(img, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 1)
    cv2.line(img, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 1)
    
    # Display elapsed time
    elapsed_time = int(time.time() - start_time)
    time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
    cv2.putText(img, time_str, (width - 150, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img

def draw_target_display(frame, x, y, w, h, color=(0, 255, 0), thickness=2):
    """Draw custom tracking target brackets and crosshair."""
    l = min(w, h) // 4  # Define bracket size
    # Draw corner brackets
    cv2.line(frame, (x, y), (x + l, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + l), color, thickness)
    cv2.line(frame, (x + w, y), (x + w - l, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + l), color, thickness)
    cv2.line(frame, (x, y + h), (x + l, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - l), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w - l, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - l), color, thickness)
    
    # Draw crosshair in the center of the object
    center_x, center_y = x + w // 2, y + h // 2
    size = min(w, h) // 4  # Define crosshair size
    cv2.line(frame, (center_x - size, center_y), (center_x + size, center_y), color, thickness)
    cv2.line(frame, (center_x, center_y - size), (center_x, center_y + size), color, thickness)

def main():
    # Load YOLO model
    model = YOLO('yolov8n.pt')  # Ensure the model file path is correct
    cap = cv2.VideoCapture('track_car.mp4')  # Ensure the video file path is correct
    start_time = time.time()  # Start time for the HUD
    tracker = cv2.TrackerKCF_create()  # Initialize KCF tracker
    tracking = False
    track_box = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no frame is captured

        # Object detection using YOLO if not tracking
        if not tracking:
            results = model(frame)  # Run YOLO inference
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    # Check if detected object is a car
                    if model.names[cls] == 'car':
                        x1, y1, x2, y2 = box.xyxy[0]
                        track_box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                        tracker.init(frame, track_box)  # Initialize the tracker with detected object
                        tracking = True  # Start tracking the car
                        break
                if tracking:
                    break

        # Update tracker if tracking
        if tracking:
            success, track_box = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in track_box]
                draw_target_display(frame, x, y, w, h)  # Draw target display on the tracked object

        # Display HUD and tracking info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_with_overlay = draw_hud(frame, fps, start_time)
        cv2.imshow('Car Tracking with HUD', frame_with_overlay)

        # Press 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
