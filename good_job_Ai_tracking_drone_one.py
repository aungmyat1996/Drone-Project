import cv2
from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
from ultralytics import YOLO
import math

# Connect to the drone (Replace with your connection string)
CONNECTION_STRING = 'udp:192.168.91.140:14550'
vehicle = connect(CONNECTION_STRING, wait_ready=True)

# Initialize YOLO model for person/car detection
model = YOLO('yolov8s.pt')

# Store selected target bounding box and flag
selected_target_bbox = None
tracking_target = False

# Define the default target type (e.g., "person" or "car")
TARGET_TYPE = 'person'  # Default target type is 'person'

# Time variables for handling hover and search mode
last_target_time = time.time()
hover_timeout = 5  # Time in seconds to hover if no target is found

# Function to check if two bounding boxes are close enough (same person or car)
def is_same_target(bbox1, bbox2, threshold=50):
    """Returns True if the two bounding boxes are close enough."""
    x1, y1, x2, y2 = bbox1
    x1_new, y1_new, x2_new, y2_new = bbox2
    center_old = ((x1 + x2) // 2, (y1 + y2) // 2)
    center_new = ((x1_new + x2_new) // 2, (y1_new + y2_new) // 2)
    distance = ((center_old[0] - center_new[0]) ** 2 + (center_old[1] - center_new[1]) ** 2) ** 0.5
    return distance < threshold

# Function to draw bounding boxes around the detected objects
def draw_bounding_boxes(frame, boxes, selected_bbox=None):
    """Draw bounding boxes and labels for detected objects."""
    for (x1, y1, x2, y2, cls_name) in boxes:
        if selected_bbox and is_same_target((x1, y1, x2, y2), selected_bbox):
            color = (0, 0, 255)  # Red for selected target
            label = "Following"
        else:
            color = (0, 255, 0)  # Green for unselected targets
            label = cls_name
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Function to handle mouse clicks and select a target
def mouse_callback(event, x, y, flags, param):
    global selected_target_bbox, tracking_target
    bounding_boxes = param
    if event == cv2.EVENT_LBUTTONDOWN:
        # Iterate through all detected bounding boxes and check if the click is inside one
        for (x1, y1, x2, y2, cls_name) in bounding_boxes:
            if x1 < x < x2 and y1 < y < y2 and cls_name == TARGET_TYPE:
                selected_target_bbox = (x1, y1, x2, y2)
                tracking_target = True
                print(f"Selected target: {cls_name} at {selected_target_bbox}")
                break

# Function to move the drone based on the object's position and size
def move_drone_based_on_position(obj_x, obj_y, obj_width, obj_height, frame_center_x, frame_center_y):
    """Move the drone based on the object's position relative to the frame center and control altitude."""
    x_offset = obj_x - frame_center_x
    y_offset = obj_y - frame_center_y

    # Calculate proportional control
    x_proportion = abs(x_offset) / frame_center_x
    y_proportion = abs(y_offset) / frame_center_y

    # Yaw control (Channel 4)
    if abs(x_offset) > 20:
        if x_offset < 0:
            vehicle.channels.overrides[4] = 1500 + int(300 * x_proportion)
            print(f"Yaw left, proportion: {x_proportion:.2f}")
        else:
            vehicle.channels.overrides[4] = 1500 - int(300 * x_proportion)
            print(f"Yaw right, proportion: {x_proportion:.2f}")
    else:
        vehicle.channels.overrides[4] = 1500
        print("Yaw steady")

    # Pitch control (Channel 2)
    if abs(y_offset) > 20:
        if y_offset < 0:
            vehicle.channels.overrides[2] = 1500 - int(300 * y_proportion)
            print(f"Move forward, proportion: {y_proportion:.2f}")
        else:
            vehicle.channels.overrides[2] = 1500 + int(300 * y_proportion)
            print(f"Move backward, proportion: {y_proportion:.2f}")
    else:
        vehicle.channels.overrides[2] = 1500
        print("Pitch steady")

    # Altitude control based on object size
    target_size = obj_width * obj_height
    altitude_adjustment = int((50000 - target_size) / 1000)  # Adjust altitude based on target size
    if altitude_adjustment > 0:
        vehicle.channels.overrides[3] = 1500 + min(altitude_adjustment, 200)
        print(f"Increasing altitude by {altitude_adjustment}")
    else:
        vehicle.channels.overrides[3] = 1500 - min(abs(altitude_adjustment), 200)
        print(f"Decreasing altitude by {altitude_adjustment}")

# Function to stop tracking and hover the drone
def stop_tracking():
    global tracking_target, selected_target_bbox
    tracking_target = False
    selected_target_bbox = None
    vehicle.channels.overrides = {
        2: 1500,  # Steady pitch
        3: 1500,  # Steady throttle (altitude)
        4: 1500   # Steady yaw
    }
    print("Stopped tracking. Hovering in place.")

# Function to create circular waypoints for auto-search
def create_search_circle(center_location, radius, num_points=8):
    """Create a circular pattern around the given center_location."""
    waypoints = []
    for i in range(num_points):
        angle = i * (360 / num_points)
        offset_x = radius * math.cos(math.radians(angle))
        offset_y = radius * math.sin(math.radians(angle))
        new_location = LocationGlobalRelative(
            center_location.lat + (offset_y / 111320),
            center_location.lon + (offset_x / (111320 * math.cos(math.radians(center_location.lat)))),
            center_location.alt
        )
        waypoints.append(new_location)
    return waypoints

# Function to start circular search pattern
def start_search_pattern():
    center_location = vehicle.location.global_relative_frame
    search_waypoints = create_search_circle(center_location, radius=10)
    for waypoint in search_waypoints:
        print(f"Moving to waypoint: {waypoint}")
        vehicle.simple_goto(waypoint)
        time.sleep(5)  # Wait for the drone to move to the next waypoint

# Function to process YOLO detections and return bounding boxes
def process_yolo_detections(frame):
    """Process YOLOv8 detections and return bounding boxes for the selected target type."""
    results = model(frame)
    detections = results[0].boxes
    bounding_boxes = []
    
    for box in detections:
        cls = int(box.cls[0].item())
        cls_name = model.names[cls]
        if cls_name == TARGET_TYPE:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            bounding_boxes.append((x1, y1, x2, y2, cls_name))
    
    return bounding_boxes

# Function to move the drone to the selected target
def move_drone_to_target(target_bbox):
    x1, y1, x2, y2 = target_bbox
    obj_x = (x1 + x2) // 2
    obj_y = (y1 + y2) // 2
    obj_width, obj_height = x2 - x1, y2 - y1
    frame_center_x, frame_center_y = 640 // 2, 480 // 2
    move_drone_based_on_position(obj_x, obj_y, obj_width, obj_height, frame_center_x, frame_center_y)

# Main function for detection and tracking
def main():
    global selected_target_bbox, TARGET_TYPE, last_target_time, tracking_target
    cap = cv2.VideoCapture('tack_car_test.mp4')  # Camera feed or video file

    # Create a window and set the mouse callback to select targets
    cv2.namedWindow("Drone Target Tracking")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection and return bounding boxes for the selected target type
        bounding_boxes = process_yolo_detections(frame)

        # Draw bounding boxes around the detected objects
        draw_bounding_boxes(frame, bounding_boxes, selected_target_bbox)

        # If a target is being tracked, move the drone towards it
        if tracking_target and selected_target_bbox:
            move_drone_to_target(selected_target_bbox)

        # Check if there is no target for hover_timeout, then hover
        if not tracking_target and time.time() - last_target_time > hover_timeout:
            stop_tracking()

        # Display the frame
        cv2.imshow("Drone Target Tracking", frame)

        # Set mouse callback for target selection
        cv2.setMouseCallback("Drone Target Tracking", mouse_callback, bounding_boxes)

        # Check for key commands
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):  # Press 'p' to track a person
            TARGET_TYPE = 'person'
            print("Tracking person")
        elif key == ord('c'):  # Press 'c' to track a car
            TARGET_TYPE = 'car'
            print("Tracking car")
        elif key == ord('s'):  # Press 's' to start the search pattern
            start_search_pattern()
        elif key == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    finally:
        vehicle.close()
