import numpy as np

left_lane_x_range = (-4.5, -1.5)
center_lane_x_range = (-1.5, 1.5)
right_lane_x_range = (1.5, 4.5)
time_interval = 1.0
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

def extract_behavior(data):
    behavior_descriptions = ""

    for car_index, car_data in enumerate(data, start=1):
        behavior_descriptions += f"Car {car_index}:\n"
        for i in range(0, len(car_data), 2):  # Iterate in pairs for 0.5-second intervals
            if car_data[i] != 'No Data' and i + 1 < len(car_data) and car_data[i + 1] != 'No Data':
                behavior_descriptions += f"  {car_data[i][0]} for 0.5 seconds\n"
                behavior_descriptions += f"  {car_data[i + 1][0]} for 0.5 seconds\n"
            elif car_data[i] != 'No Data':
                behavior_descriptions += f"  {car_data[i][0]} for 0.5 seconds\n"
            elif i + 1 < len(car_data) and car_data[i + 1] != 'No Data':
                behavior_descriptions += f"  {car_data[i + 1][0]} for 0.5 seconds\n"

    return behavior_descriptions.strip()