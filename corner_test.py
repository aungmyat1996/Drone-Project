import pygame
import sys

# Initialize Pygame
pygame.init()

# Set the display
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Drone Tracking UI")

# Set colors
black = (0, 0, 0)
green = (0, 255, 0)
red = (255, 0, 0)

# Corner length (the size of each corner line)
corner_length = 20

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Fill the background
    screen.fill(black)
     # Draw central crosshair
#     pygame.draw.line(screen, green, (400, 290), (400, 310), 2)  # Vertical line
#     pygame.draw.line(screen, green, (390, 300), (410, 300), 2)  # Horizontal line
    
    # Draw corners
    # Top left corner
    pygame.draw.line(screen, green, (190, 210), (190 + corner_length, 210), 2)  # Top horizontal line
    pygame.draw.line(screen, green, (190, 210), (190, 210 + corner_length), 2)  # Left vertical line
    # Top right corner
    pygame.draw.line(screen, green, (615 - corner_length, 210), (615, 210), 2)  # Top horizontal line
    pygame.draw.line(screen, green, (615, 210), (615, 210 + corner_length), 2)  # Right vertical line
    # Bottom left corner
    pygame.draw.line(screen, green, (190, 410), (190 + corner_length, 410), 2)  # Bottom horizontal line
    pygame.draw.line(screen, green, (190, 410 - corner_length), (190, 410), 2)  # Left vertical line
    # Bottom right corner
    pygame.draw.line(screen, green, (615 - corner_length, 410), (615, 410), 2)  # Bottom horizontal line
    pygame.draw.line(screen, green, (615, 410 - corner_length), (615, 410), 2)  # Right vertical line
    # Draw recording indicator
    pygame.draw.circle(screen, red, (400, 304), 5)
    # Draw central crosshair
    pygame.draw.line(screen, green, (350, 300), (385, 300), 1)
    pygame.draw.line(screen, green, (413, 300), (450, 300), 1)
    pygame.draw.line(screen, green, (400, 315), (400, 340), 1)
    
    
    # Update display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
