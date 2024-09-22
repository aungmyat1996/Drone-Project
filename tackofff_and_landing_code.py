from dronekit import connect, VehicleMode
import time

# Connect to the drone (UDP connection)
CONNECTION_STRING = 'udp:192.168.91.140:14550'
vehicle = connect(CONNECTION_STRING, wait_ready=True)

# Function to arm the drone and take off to a specific altitude in Guided Mode
def arm_and_takeoff(target_altitude=10):
    print("Arming motors and taking off to search altitude...")
    while not vehicle.is_armable:
        print("Waiting for vehicle to initialize...")
        time.sleep(1)

    vehicle.armed = True
    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(1)

    print(f"Taking off to {target_altitude} meters!")
    vehicle.simple_takeoff(target_altitude)

    # Wait until the drone reaches the target altitude
    while True:
        current_altitude = vehicle.location.global_relative_frame.alt
        print(f"Altitude: {current_altitude:.2f} meters")
        if current_altitude >= target_altitude * 0.95:
            print(f"Reached target altitude: {target_altitude} meters")
            break
        time.sleep(1)

# Function to land the drone after 1 minute of hovering
def land_after_delay(delay_seconds=60):
    print(f"Hovering for {delay_seconds / 60} minute(s)...")
    time.sleep(delay_seconds)  # Wait for 1 minute

    print("Landing now...")
    vehicle.mode = VehicleMode("LAND")
    while vehicle.mode != "LAND":
        print("Waiting for drone to enter LAND mode...")
        time.sleep(1)

    # Monitor altitude until the drone lands
    while vehicle.location.global_relative_frame.alt > 0.1:
        print(f"Altitude: {vehicle.location.global_relative_frame.alt:.2f} meters - Landing...")
        time.sleep(1)

    print("Landed successfully.")

# Function to wait until Guided Mode is manually set
def wait_for_guided_mode():
    print("Waiting for Guided Mode to be set...")

    while vehicle.mode.name != "GUIDED":
        print(f"Current Mode: {vehicle.mode.name} - Waiting for Guided Mode...")
        time.sleep(1)

    print("Guided Mode detected! Proceeding with takeoff...")

# Main function to handle the process
def main():
    # Step 1: Wait for the user to set Guided Mode
    wait_for_guided_mode()

    # Step 2: Takeoff to 10 meters once in Guided Mode
    arm_and_takeoff(10)

    # Step 3: Wait for 1 minute, then land the drone
    land_after_delay(60)

if __name__ == "__main__":
    try:
        main()
    finally:
        print("Closing vehicle connection.")
        vehicle.close()
