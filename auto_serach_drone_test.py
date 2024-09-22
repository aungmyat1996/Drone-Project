from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math

# Connect to the drone (UDP connection)
vehicle = connect('udp:192.168.91.140:14550', wait_ready=True)

# Function to arm the drone and take off
def arm_and_takeoff(target_altitude=30):
    print("Pre-Flight checks")
    while not vehicle.is_armable:
        print("Waiting for vehicle to initialize...")
        time.sleep(1)

    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(1)

    print(f"Taking off to {target_altitude} meters!")
    vehicle.simple_takeoff(target_altitude)

    while True:
        print(f"Altitude: {vehicle.location.global_relative_frame.alt:.2f} meters")
        if vehicle.location.global_relative_frame.alt >= target_altitude * 0.95:
            print(f"Reached target altitude: {target_altitude} meters")
            break
        time.sleep(1)

    # Once altitude is reached, begin search
    print("Starting auto-search for targets...")

# Function to simulate drone searching by flying in a circular pattern
def perform_auto_search(search_radius=20, altitude=30, duration=60):
    """
    Perform a circular search pattern around the drone's current location.
    :param search_radius: The radius of the circular search in meters.
    :param altitude: The altitude of the search.
    :param duration: The duration to perform the search (in seconds).
    """
    # Get the drone's current location
    current_location = vehicle.location.global_relative_frame
    home_lat = current_location.lat
    home_lon = current_location.lon

    print("Starting circular search pattern...")
    start_time = time.time()

    while time.time() - start_time < duration:
        for angle in range(0, 360, 15):  # Increment angle by 15 degrees
            # Calculate next waypoint using basic trigonometry for circular path
            next_lat = home_lat + (search_radius / 111320) * math.cos(math.radians(angle))
            next_lon = home_lon + (search_radius / (111320 * math.cos(math.radians(home_lat)))) * math.sin(math.radians(angle))

            # Move the drone to the next waypoint
            target_location = LocationGlobalRelative(next_lat, next_lon, altitude)
            print(f"Moving to: {next_lat:.6f}, {next_lon:.6f} at {altitude}m")
            vehicle.simple_goto(target_location)

            # Wait for a few seconds to simulate scanning at this position
            time.sleep(5)

    print("Completed auto-search pattern.")

# Main function
def main():
    # Takeoff to a height of 30 meters for auto-search
    arm_and_takeoff(30)

    # Perform a search pattern in a radius of 20 meters for 60 seconds
    perform_auto_search(search_radius=20, altitude=30, duration=60)

    # Once the search is complete, you can land the drone or perform further actions
    print("Search complete, returning to launch.")
    vehicle.mode = VehicleMode("RTL")  # Return to Launch mode

    # Wait until the vehicle lands before closing the connection
    while vehicle.location.global_relative_frame.alt > 1:
        print(f"Current Altitude: {vehicle.location.global_relative_frame.alt:.2f} meters")
        time.sleep(1)

    print("Landed.")

if __name__ == "__main__":
    try:
        main()
    finally:
        print("Closing vehicle connection")
        vehicle.close()
