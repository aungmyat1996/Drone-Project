from ultralytics import YOLO
import cv2
import cvzone
import math

# Open video file or camera feed
cap = cv2.VideoCapture(r"C:\Users\Aung-Myat\Videos\object detection\ppe-3-1.mp4")

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Video file could not be opened or found.")
    exit(1)

# Load YOLO model
model = YOLO("ppe.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

while True:
    success, img = cap.read()

    # Check if the frame has been successfully read
    if not success:
        print("Warning: No more frames to read or failed to read the frame.")
        break  # Break out of the loop if there are no more frames

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if conf > 0.5:
                color = (0, 0, 255) if 'NO-' in currentClass else (0, 255, 0)
                cvzone.putTextRect(img, f'{currentClass} {conf}', (x1, max(35, y1)), scale=1, thickness=1, colorB=color, colorT=(255, 255, 255), colorR=color, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()
