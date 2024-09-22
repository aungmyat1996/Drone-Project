import cv2
import numpy as np

def draw_hud(frame):
    h, w = frame.shape[:2]

    # Define the lengths and offsets as proportions of the frame dimensions
    bracket_len = int(min(w, h) * 0.05)  # 5% of the shortest dimension
    thickness = 2
    offset = int(min(w, h) * 0.3)  # 2% of the shortest dimension

    # Draw corner brackets on the frame
    # Top-left
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

    # Optional: Text overlay for telemetry
    cv2.putText(frame, "FPS: 30", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "Flight Time: 00:00:01", (w - 400, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "Battery", (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "GPS: 24", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "Speed: 35 km/h", (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "GUIDED", (w - 80, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "Altitude: 100", (w - 120, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def main():
    # Capture from video or webcam
    cap = cv2.VideoCapture(0)  # Replace 'path_to_video.mp4' with 0 for webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw the HUD
        draw_hud(frame)
        
        cv2.imshow('FPV View', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
