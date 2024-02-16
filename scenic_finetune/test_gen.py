import numpy as np
import os
import json

# Define lane ranges
left_lane_x_range = (-4.5, -1.5)
center_lane_x_range = (-1.5, 1.5)
right_lane_x_range = (1.5, 4.5)
time_interval = 0.5
# Each frame is 0.5 seconds

# Map lane ranges for easy reference
lane_ranges = {
    "left": left_lane_x_range,
    "center": center_lane_x_range,
    "right": right_lane_x_range
}


# Function to calculate distance between two coordinates
def calculate_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


# Determine current lane based on x-coordinate (relative to starting x)
def get_lane_id(x, start_x):
    adjusted_x = x - start_x
    for lane_id, (low, high) in lane_ranges.items():
        if low <= adjusted_x <= high:
            return lane_id
    return None


# Interpret car actions based on lane and speed, relative to initial frame
def interpret_action_with_speed(car_coords, time_interval=0.5):
    all_actions = []

    for car in car_coords:
        # Find the first valid coordinate to use as a reference (start_x)
        start_x = None
        for coord in car:
            if coord is not None:
                start_x = coord[0]
                break

        if start_x is None:
            all_actions.append("No valid data")  # Skip cars with no data
            continue

        car_actions = []

        for i in range(1, len(car)):
            # Skip frames where data is missing (None)
            if car[i] is None or car[i - 1] is None:
                car_actions.append("No Data")
                continue

            x, y = car[i]
            prev_x, prev_y = car[i - 1]

            # Calculate speed
            distance = calculate_distance((x, y), (prev_x, prev_y))
            speed = distance / time_interval

            # Determine lane and action, using adjusted x-coordinate relative to start_x
            lane_id = get_lane_id(x, start_x)
            prev_lane_id = get_lane_id(prev_x, start_x)

            if lane_id is None:
                action = f"Driving Independently at speed {speed:.2f}"
            elif lane_id == prev_lane_id:
                action = f"Follows Lane at speed {speed:.2f}"
            elif lane_id != prev_lane_id:
                direction = "right" if lane_id == "right" or (
                            lane_id == "center" and prev_lane_id == "left") else "left"
                action = f"Changes Lane to the {direction} at speed {speed:.2f}"

            car_actions.append((action, speed))

        all_actions.append(car_actions)

    return all_actions

