import cv2
from dronekit import connect, VehicleMode
import time
from ultralytics import YOLO  # YOLO for object detection

# Connect to the drone (UDP connection)
CONNECTION_STRING = 'udp:192.168.91.140:14550'
vehicle = connect(CONNECTION_STRING, wait_ready=True)

# Initialize YOLO model for target detection
model = YOLO('yolov8s.pt')  # Make sure you have the YOLOv8s model weights

# Set the desired target class (person or car)
TARGET_CLASS = 'person'  # You can change this to 'car' or any YOLO class name

# Function to arm and takeoff the drone
def arm_and_takeoff(target_altitude=10):
    print("Arming motors and taking off...")
    
    # Wait for the vehicle to be armable
    while not vehicle.is_armable:
        print("Waiting for vehicle to initialize...")
        time.sleep(1)
    
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    
    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(1)
    
    print(f"Taking off to {target_altitude} meters!")
    vehicle.simple_takeoff(target_altitude)
    
    # Wait until the vehicle reaches a safe altitude
    while True:
        current_altitude = vehicle.location.global_relative_frame.alt
        print(f"Altitude: {current_altitude:.2f} meters")
        if current_altitude >= target_altitude * 0.95:
            print(f"Reached target altitude: {target_altitude} meters")
            break
        time.sleep(1)

# Function to move the drone based on the target's position in the frame
def move_drone_based_on_position(obj_x, obj_y, frame_center_x, frame_center_y):
    """Move the drone based on the object's position in the frame."""
    x_offset = obj_x - frame_center_x
    y_offset = obj_y - frame_center_y
    
    x_threshold = 20  # Threshold for horizontal movement
    y_threshold = 20  # Threshold for vertical movement

    # Yaw (left-right) control
    if abs(x_offset) > x_threshold:
        if x_offset < 0:  # Target is to the left
            vehicle.channels.overrides[4] = 1600  # Yaw left
        else:  # Target is to the right
            vehicle.channels.overrides[4] = 1400  # Yaw right
    else:
        vehicle.channels.overrides[4] = 1500  # Keep yaw steady

    # Pitch (forward-backward) control
    if abs(y_offset) > y_threshold:
        if y_offset < 0:  # Target is above the center
            vehicle.channels.overrides[2] = 1600  # Move forward
        else:  # Target is below the center
            vehicle.channels.overrides[2] = 1400  # Move backward
    else:
        vehicle.channels.overrides[2] = 1500  # Keep pitch steady

# Function to process YOLO detections and select the target
def process_yolo_detections(frame):
    results = model(frame)
    detections = results[0].boxes
    target_found = False

    # Loop through detected objects
    for box in detections:
        cls = int(box.cls[0].item())
        if model.names[cls] == TARGET_CLASS:  # Check if the target class matches
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Get bounding box
            obj_x = (x1 + x2) // 2
            obj_y = (y1 + y2) // 2

            # Draw bounding box and label around the detected target
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, TARGET_CLASS, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Move drone based on the target's position in the frame
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            move_drone_based_on_position(obj_x, obj_y, frame_center_x, frame_center_y)

            target_found = True

    return target_found

# Main function to handle drone tracking and detection
def main():
    print("Waiting for Guided Mode...")
    
    # Wait for the drone to be in Guided Mode
    while vehicle.mode.name != "GUIDED":
        time.sleep(1)
    
    # Takeoff to 10 meters
    arm_and_takeoff(10)
    
    # Initialize video capture (webcam or video file)
    cap = cv2.VideoCapture('test_56.mp4')  # Use 0 for webcam, or a video file path

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process YOLO detections
        target_found = process_yolo_detections(frame)
        
        if not target_found:
            print("Target not found. Hovering...")
            vehicle.channels.overrides = {'2': 1500, '4': 1500}  # Keep drone hovering steady

        # Display the video feed
        cv2.imshow("Drone Target Tracking", frame)

        # Check if 'q' is pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    try:
        main()
    finally:
        print("Closing vehicle connection...")
        vehicle.close()
