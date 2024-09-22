import cv2
import numpy as np
import time

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def draw_hud(frame, start_time, frame_count):
    h, w = frame.shape[:2]
    current_time = time.time()
    fps = frame_count / (current_time - start_time) if current_time > start_time else 0

    # Define the lengths and offsets as proportions of the frame dimensions
    bracket_len = int(min(w, h) * 0.05)  # 5% of the shortest dimension
    thickness = 2
    offset = int(min(w, h) * 0.1)  # 10% of the shortest dimension

    # Draw corner brackets pointing inward (opposite direction)
    # Top-left (brackets pointing toward the center)
    cv2.line(frame, (offset, offset), (offset + bracket_len, offset), (0, 255, 0), thickness)
    cv2.line(frame, (offset, offset), (offset, offset + bracket_len), (0, 255, 0), thickness)

    # Top-right
    cv2.line(frame, (w - offset, offset), (w - offset - bracket_len, offset), (0, 255, 0), thickness)
    cv2.line(frame, (w - offset, offset), (w - offset, offset + bracket_len), (0, 255, 0), thickness)

    # Bottom-left
    cv2.line(frame, (offset, h - offset), (offset + bracket_len, h - offset), (0, 255, 0), thickness)
    cv2.line(frame, (offset, h - offset), (offset, h - offset - bracket_len), (0, 255, 0), thickness)

    # Bottom-right
    cv2.line(frame, (w - offset, h - offset), (w - offset - bracket_len, h - offset), (0, 255, 0), thickness)
    cv2.line(frame, (w - offset, h - offset), (w - offset, h - offset - bracket_len), (0, 255, 0), thickness)

    # Telemetry text overlays
    text_props = {"fontFace": cv2.FONT_HERSHEY_SIMPLEX, "fontScale": 0.5, "color": (0, 255, 0), "thickness": 1}
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 180), **text_props)
    cv2.putText(frame, "Flight Time: 00:00:01", (20, 200), **text_props)
    cv2.putText(frame, "Battery", (20, 220), **text_props)
    cv2.putText(frame, "GPS: 24", (20, 240), **text_props)
    cv2.putText(frame, "Speed: 35 km/h", (20, 260), **text_props)
    cv2.putText(frame, "GUIDED", (20, 280), **text_props)
    cv2.putText(frame, "Altitude: 100", (20, 300), **text_props)

def detect_and_draw_faces(frame):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera or video file could not be opened.")
        return

    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        # Detect and draw faces
        detect_and_draw_faces(frame)

        # Draw the HUD
        draw_hud(frame, start_time, frame_count)
        
        # Show the frame with HUD and face tracking
        cv2.imshow('FPV View with Face Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
