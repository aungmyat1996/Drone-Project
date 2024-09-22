import cv2
import numpy as np

def draw_bounding_box(image):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Define bounding box parameters
    box_width = int(width * 0.4)  # 40% of image width
    box_height = int(height * 0.4)  # 40% of image height
    center_x = width // 2
    center_y = height // 2
    
    # Calculate corner coordinates
    top_left = (center_x - box_width // 2, center_y - box_height // 2)
    bottom_right = (center_x + box_width // 2, center_y + box_height // 2)
    
    # Draw main rectangle
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 1)
    
    # Draw corner markers
    marker_length = 1
    corners = [top_left, 
               (bottom_right[0], top_left[1]),
               (top_left[0], bottom_right[1]),
               bottom_right]
    
    for corner in corners:
        x, y = corner
        # Horizontal line
        cv2.line(image, (x - marker_length, y), (x + marker_length, y), (0, 255, 0), 1)
        # Vertical line
        cv2.line(image, (x, y - marker_length), (x, y + marker_length), (0, 255, 0), 1)
    
    # Draw crosshair
    cv2.line(image, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 0), 1)
    cv2.line(image, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 0), 1)
    
    return image

# Load an image (replace 'path_to_your_image.jpg' with your image path)
img = cv2.imread('test.jpg')

# Apply the bounding box
result = draw_bounding_box(img)

# Display the result
cv2.imshow('Bounding Box', result)
cv2.waitKey(0)
cv2.destroyAllWindows()