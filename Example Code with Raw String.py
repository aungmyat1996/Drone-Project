from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8l.pt')  # Load the YOLOv8 model

# Perform inference on the image
#results = model(r"C:\Users\Aung-Myat\Videos\object detection\people.mp4", show=True)
results = model("C:\\Users\\Aung-Myat\\Videos\\object detection\\test.jpg", show=True)
#results = model(r"C:\Users\Aung-Myat\Videos\object detection\test.jpg", show=True)



# Wait for a key press to close the OpenCV window
cv2.waitKey(0)
