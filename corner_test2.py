import pygame
import sys

# Initialize Pygame
pygame.init()

# Set the display size
screen_width = 640
screen_height = 480
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Drone Tracking UI")

# Set colors
black = (0, 0, 0)
green = (0, 255, 0)
red = (255, 0, 0)

# Corner length (the size of each corner line) - reduced for smaller frame
corner_length = 20  # Reduced from 20 to 15

# Margin from the edges of the window
margin = 190  # Reduced or adjusted as needed to change the placement

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Fill the background
    screen.fill(black)
    
    # Calculate positions for stable corner placement
    bottom_y = screen_height - margin
    right_x = screen_width - margin
    center_x, center_y = screen_width // 2, screen_height // 2
    
    # Draw corners
    # Top left corner
    pygame.draw.line(screen, green, (margin, margin), (margin + corner_length, margin), 2)  # Top horizontal line
    pygame.draw.line(screen, green, (margin, margin), (margin, margin + corner_length), 2)  # Left vertical line
    
    # Top right corner
    pygame.draw.line(screen, green, (right_x - corner_length, margin), (right_x, margin), 2)  # Top horizontal line
    pygame.draw.line(screen, green, (right_x, margin), (right_x, margin + corner_length), 2)  # Right vertical line
    
    # Bottom left corner
    pygame.draw.line(screen, green, (margin, bottom_y), (margin + corner_length, bottom_y), 2)  # Bottom horizontal line
    pygame.draw.line(screen, green, (margin, bottom_y - corner_length), (margin, bottom_y), 2)  # Left vertical line
    
    # Bottom right corner
    pygame.draw.line(screen, green, (right_x - corner_length, bottom_y), (right_x, bottom_y), 2)  # Bottom horizontal line
    pygame.draw.line(screen, green, (right_x, bottom_y - corner_length), (right_x, bottom_y), 2)  # Right vertical line
    
    # Draw central crosshair
    pygame.draw.line(screen, green, (center_x - 30, center_y), (center_x + 30, center_y), 1)  # Horizontal line
    pygame.draw.line(screen, green, (center_x, center_y - 25), (center_x, center_y + 25), 1)  # Vertical line
    
    # Update display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
