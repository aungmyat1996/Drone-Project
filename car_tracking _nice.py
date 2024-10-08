import cv2
from ultralytics import YOLO
import time

# Initialize YOLOv8s model for car detection
model = YOLO('yolov8s.pt')

# Initialize video capture (use 0 for webcam or path to video file)
cap = cv2.VideoCapture('track_car.mp4')

# Initialize tracker (we'll use CSRT for more accurate tracking)
trackers = []
tracking = False

# Start time and frame count for FPS calculation
start_time = time.time()
frame_count = 0

# Initialize the car count and status
car_count = 0
last_detection_time = 0
detection_interval = 2  # YOLO detection every 2 seconds

def calculate_fps(start_time, frame_count):
    """ Calculate and update FPS. """
    current_time = time.time()
    fps = frame_count / (current_time - start_time) if current_time > start_time else 0
    return fps

def draw_corner_brackets(frame):
    """ Draw corner brackets on the frame. """
    h, w = frame.shape[:2]
    bracket_len = int(min(w, h) * 0.1)  # 5% of the shortest dimension
    thickness = 2
    offset = int(min(w, h) * 0.27)  # 10% offset from the edges

    # Top-left corner
    cv2.line(frame, (offset, offset), (offset + bracket_len, offset), (0, 255, 0), thickness)
    cv2.line(frame, (offset, offset), (offset, offset + bracket_len), (0, 255, 0), thickness)

    # Top-right corner
    cv2.line(frame, (w - offset, offset), (w - offset - bracket_len, offset), (0, 255, 0), thickness)
    cv2.line(frame, (w - offset, offset), (w - offset, offset + bracket_len), (0, 255, 0), thickness)

    # Bottom-left corner
    cv2.line(frame, (offset, h - offset), (offset + bracket_len, h - offset), (0, 255, 0), thickness)
    cv2.line(frame, (offset, h - offset), (offset, h - offset - bracket_len), (0, 255, 0), thickness)

    # Bottom-right corner
    cv2.line(frame, (w - offset, h - offset), (w - offset - bracket_len, h - offset), (0, 255, 0), thickness)
    cv2.line(frame, (w - offset, h - offset), (w - offset, h - offset - bracket_len), (0, 255, 0), thickness)
    
def draw_hud(frame, fps, car_count, elapsed_time):
    """ Draw HUD with FPS and other telemetry. """
    h, w = frame.shape[:2]
    text_props = {"fontFace": cv2.FONT_HERSHEY_SIMPLEX, "fontScale": 0.5, "color": (0, 255, 0), "thickness": 2}
    
    # Telemetry info
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30), **text_props)
    cv2.putText(frame, f"Car Count: {car_count}", (20, 70), **text_props)
    cv2.putText(frame, f"Elapsed Time: {elapsed_time:.2f}s", (20, 110), **text_props)

    # Add more telemetry data as needed
    cv2.putText(frame, "Speed: 45 km/h", (w - 200, 30), **text_props)
    cv2.putText(frame, "Altitude: 200 m", (w - 200, 70), **text_props)
    cv2.putText(frame, "Battery: 80%", (w - 200, 110), **text_props)

def process_yolo_detections(frame, results):
    """ Detect cars using YOLOv8 and initialize trackers for each detected car. """
    global trackers, tracking, car_count
    trackers = []  # Reset the list of trackers
    car_count = 0  # Reset car count

    try:
        detections = results[0].boxes  # Access bounding boxes from YOLO results
    except AttributeError:
        print("Error: Unable to access 'boxes'.")
        return

    # Loop through the detected boxes
    for box in detections:
        cls = int(box.cls[0].item())  # Class label
        if model.names[cls] == 'car':  # Check if the detected object is a car
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Extract bounding box coordinates

            # Initialize a CSRT tracker for each detected car
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            trackers.append(tracker)
            car_count += 1  # Increment car count
    
    tracking = len(trackers) > 0  # Enable tracking if trackers are available

def update_trackers(frame):
    """ Update all trackers and draw bounding boxes for tracked cars. """
    global trackers
    tracked_cars = 0
    for tracker in trackers:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Car", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            tracked_cars += 1
    return tracked_cars

def main():
    global tracking, last_detection_time
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.time()

        # Run YOLO inference every detection_interval seconds or when no tracking is active
        if not tracking or (current_time - last_detection_time) > detection_interval:
            results = model(frame)
            process_yolo_detections(frame, results)
            last_detection_time = current_time

        # Update all trackers
        if tracking:
            tracked_cars = update_trackers(frame)

        # Calculate FPS and elapsed time
        fps = calculate_fps(start_time, frame_count)
        elapsed_time = current_time - start_time

        # Draw the corner brackets
        draw_corner_brackets(frame)

        # Draw the HUD (telemetry, FPS, car count, etc.)
        draw_hud(frame, fps, car_count, elapsed_time)

        # Display the frame with tracking and HUD
        cv2.imshow('YOLOv8 Car Tracking with Enhanced UI and Corner Brackets', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
