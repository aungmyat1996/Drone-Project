import cv2
from ultralytics import YOLO
import time
from dronekit import connect, VehicleMode

# Initialize YOLOv8 model for car/person detection
model = YOLO('yolov8s.pt')

# Connect to SITL or real drone
CONNECTION_STRING = 'udp:192.168.91.140:14551'
vehicle = connect(CONNECTION_STRING, wait_ready=True)

# Function to arm the drone and take off
def arm_and_takeoff(target_altitude):
    print("Arming motors")
    while not vehicle.is_armable:
        print("Waiting for vehicle to initialize...")
        time.sleep(1)

    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(target_altitude)

    while True:
        print(f"Altitude: {vehicle.location.global_relative_frame.alt}")
        if vehicle.location.global_relative_frame.alt >= target_altitude * 0.95:
            print(f"Reached target altitude: {target_altitude}m")
            break
        time.sleep(1)

# Initialize video capture (0 for webcam, or path to video file)
cap = cv2.VideoCapture('track_car.mp4')

# Drone control and tracking variables
AUTO_TRACKING = False  # Switch between manual and auto modes
DRONE_SPEED = 10  # Adjusted speed for smoother drone movement

# Tracking flags
trackers = []
tracking = False
car_count = 0
last_detection_time = 0
detection_interval = 2  # YOLO detection interval (in seconds)

# FPS and timing
start_time = time.time()
frame_count = 0

# Takeoff the drone to 10 meters altitude
arm_and_takeoff(10)

def calculate_fps(start_time, frame_count):
    current_time = time.time()
    fps = frame_count / (current_time - start_time) if current_time > start_time else 0
    return fps

def draw_hud(frame, fps, car_count, elapsed_time):
    """Draw HUD data such as FPS, car count, and mode."""
    h, w = frame.shape[:2]
    text_props = {"fontFace": cv2.FONT_HERSHEY_SIMPLEX, "fontScale": 0.7, "color": (0, 255, 0), "thickness": 2}
    
    # Telemetry information
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30), **text_props)
    cv2.putText(frame, f"Car Count: {car_count}", (20, 70), **text_props)
    cv2.putText(frame, f"Elapsed Time: {elapsed_time:.2f}s", (20, 110), **text_props)
    cv2.putText(frame, "Mode: " + ("Auto" if AUTO_TRACKING else "Manual"), (20, 150), **text_props)

def switch_mode():
    """Switch between manual and auto-tracking modes."""
    global AUTO_TRACKING
    AUTO_TRACKING = not AUTO_TRACKING
    print(f"Switched to {'Auto-Tracking' if AUTO_TRACKING else 'Manual'} Mode")

def process_yolo_detections(frame, results):
    """Process YOLO detections to initialize tracking."""
    global trackers, tracking, car_count
    trackers = []
    car_count = 0

    try:
        detections = results[0].boxes
    except AttributeError:
        print("Error: No bounding boxes detected.")
        return

    # Track detected cars
    for box in detections:
        cls = int(box.cls[0].item())
        if model.names[cls] == 'car':
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            trackers.append(tracker)
            car_count += 1

    tracking = len(trackers) > 0

def update_trackers(frame):
    """Update car trackers and send drone movement commands."""
    global trackers
    for tracker in trackers:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Car", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if AUTO_TRACKING:
                move_drone_based_on_position(x + w // 2, y + h // 2, frame.shape[1] // 2, frame.shape[0] // 2)

def move_drone_based_on_position(obj_x, obj_y, center_x, center_y):
    """Move the drone based on the object's position in the frame."""
    
    # Adjust yaw (left-right) based on horizontal position of the car (Channel 4)
    if obj_x < center_x - 50:  # Object is left of center
        vehicle.channels.overrides[4] = 1500 + DRONE_SPEED
        print("Moving left (yaw)")
    elif obj_x > center_x + 50:  # Object is right of center
        vehicle.channels.overrides[4] = 1500 - DRONE_SPEED
        print("Moving right (yaw)")
    else:
        vehicle.channels.overrides[4] = 1500  # Keep yaw steady
        print("Yaw steady")

    # Adjust pitch (forward-backward) based on vertical position of the car (Channel 2)
    if obj_y < center_y - 50:  # Object is above center
        vehicle.channels.overrides[2] = 1500 - DRONE_SPEED
        print("Moving forward (pitch)")
    elif obj_y > center_y + 50:  # Object is below center
        vehicle.channels.overrides[2] = 1500 + DRONE_SPEED
        print("Moving backward (pitch)")
    else:
        vehicle.channels.overrides[2] = 1500  # Keep pitch steady
        print("Pitch steady")

# Main loop for drone control and tracking
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()

    # Run YOLO detection every 2 seconds
    if not tracking or (current_time - last_detection_time) > detection_interval:
        results = model(frame)
        process_yolo_detections(frame, results)
        last_detection_time = current_time

    # Update the trackers and send drone commands
    if tracking:
        update_trackers(frame)

    # Calculate FPS and draw HUD
    fps = calculate_fps(start_time, frame_count)
    elapsed_time = current_time - start_time
    draw_hud(frame, fps, car_count, elapsed_time)

    cv2.imshow('YOLOv8 Car Tracking with Drone', frame)

    # Switch modes with 'm' key, quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('m'):
        switch_mode()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
