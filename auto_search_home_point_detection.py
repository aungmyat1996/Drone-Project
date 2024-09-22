from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math

# Connect to the drone (Replace with your connection string)
CONNECTION_STRING = 'udp:192.168.91.140:14550'
try:
    vehicle = connect(CONNECTION_STRING, wait_ready=True)
except Exception as e:
    print(f"Error connecting to vehicle: {e}")
    exit()

def arm_and_takeoff(target_altitude):
    """
    Arms vehicle and flies to a target altitude.
    """
    print("Basic pre-arm checks...")
    # Don't try to arm until autopilot is ready
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors...")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    # Confirm vehicle armed before attempting to takeoff
    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print(f"Taking off to {target_altitude} meters!")
    vehicle.simple_takeoff(target_altitude)  # Takeoff to target altitude

    # Wait until the vehicle reaches a safe altitude
    while True:
        print(f" Altitude: {vehicle.location.global_relative_frame.alt}")
        if vehicle.location.global_relative_frame.alt >= target_altitude * 0.95:
            print(f"Reached target altitude: {target_altitude} meters")
            break
        time.sleep(1)

def create_search_circle(center_location, radius, num_points=8, altitude=10):
    """
    Create a circular pattern around the given center_location with a safe altitude.
    """
    waypoints = []
    for i in range(num_points):
        angle = i * (360 / num_points)
        offset_x = radius * math.cos(math.radians(angle))
        offset_y = radius * math.sin(math.radians(angle))
        new_location = LocationGlobalRelative(
            center_location.lat + (offset_y / 111320),  # Latitude offset
            center_location.lon + (offset_x / (111320 * math.cos(math.radians(center_location.lat)))),  # Longitude offset
            altitude  # Set a fixed positive altitude
        )
        waypoints.append(new_location)
    return waypoints

def start_search_pattern():
    """
    Initiate a circular search pattern when no target is found.
    """
    # Make sure the drone has taken off before initiating the search
    if vehicle.location.global_relative_frame.alt < 2:  # Assuming below 2 meters means on the ground
        print("Taking off before starting search pattern...")
        arm_and_takeoff(10)  # Take off to 10 meters

    center_location = vehicle.location.global_relative_frame
    search_waypoints = create_search_circle(center_location, radius=10, altitude=10)  # Set a safe altitude
    
    for waypoint in search_waypoints:
        print(f"Moving to waypoint: {waypoint}")
        vehicle.simple_goto(waypoint)
        time.sleep(5)  # Pause to allow the drone to reach each waypoint

    print("Search pattern complete.")

def main():
    try:
        # Arm and take off to a safe altitude before beginning search
        arm_and_takeoff(10)  # Takeoff to 10 meters
        
        # Start the auto-search pattern
        start_search_pattern()
        
    except KeyboardInterrupt:
        print("User interrupted execution.")
    finally:
        # Close the vehicle connection
        print("Returning vehicle to home...")
        vehicle.mode = VehicleMode("RTL")
        vehicle.close()

if __name__ == "__main__":
    main()
