def generate_behavior_code(car_behaviors):
    behavior_scripts = []

    for car_index, behaviors in enumerate(car_behaviors):
        car_name = f"Behavior_car{car_index + 1}()"
        behavior_script = [f"behavior {car_name}:"]
        last_behavior = None
        last_speed = None
        current_duration = 0

        for i, behavior in enumerate(behaviors):
            if behavior == 'No Data':
                continue

            current_behavior, speed = behavior  # Extract behavior type and speed

            if "Follows Lane" in current_behavior:
                # If continuing with the same speed, accumulate the duration
                if last_behavior == "Follows Lane" and last_speed == speed:
                    current_duration += 0.5
                else:
                    # If speed changes or a new "Following Lane" starts, flush the last behavior
                    if last_behavior == "Follows Lane" and current_duration > 0:
                        behavior_script.append(f"    do CustomFollowLaneBehavior({last_speed:.2f}) for {current_duration:.1f} seconds")
                    current_duration = 0.5  # Start counting for the new follow behavior

                last_behavior = "Follows Lane"
                last_speed = speed

            elif "Changing to" in current_behavior:
                # Flush any accumulated "Following Lane" behavior before adding the lane change
                if last_behavior == "Following Lane" and current_duration > 0:
                    behavior_script.append(f"    do CustomFollowLaneBehavior({last_speed:.2f}) for {current_duration:.1f} seconds")
                    current_duration = 0

                # Extract lane change direction (left/right) from the behavior string
                direction = "Left" if "Left" in current_behavior else "Right"
                behavior_script.append(f"    do CustomLaneChangeBehavior(network.laneSectionAt(self)._laneTo{direction})")
                last_behavior = "Changes Lane"
                last_speed = None  # Reset speed after lane change

        # Append any remaining "Following Lane" behavior at the end
        if last_behavior == "Follows Lane" and current_duration > 0:
            behavior_script.append(f"    do CustomFollowLaneBehavior({last_speed:.2f}) for {current_duration:.1f} seconds")

        if len(behavior_script) > 1:  # Ensure there is actual behavior content
            behavior_scripts.append("\n".join(behavior_script))

    return "\n\n".join(behavior_scripts)


def print_car_behavior(car_data):
    """
    Processes and prints the behavior of each car at each frame, skipping frames with 'No Data'.

    Parameters:
    car_data (list): A nested list where each sublist represents a car's data,
                     and each element in the sublist represents data for each frame.
    """
    for car_index, car_frames in enumerate(car_data, start=1):
        print(f"Car {car_index}:")
        has_data = False  # Flag to check if there is any valid data

        for frame_index, frame_data in enumerate(car_frames, start=1):
            if frame_data != 'No Data':
                description, speed = frame_data
                print(f"  Frame {frame_index}: {description} at speed {speed:.2f} m/s")
                has_data = True

        if not has_data:
            print("  No valid data available for this car.")

        print()  # Add a newline for better separation between cars
# Example input
car_behaviors = [
    [('Following Lane at speed 1.52', 1.5150439883885094), 'No Data', 'No Data', 'No Data'],
    [('Following Lane at speed 0.09', 0.09191086377849372), ('Following Lane at speed 0.25', 0.25379590746897307), 'No Data', 'No Data'],
    [('Following Lane at speed 0.54', 0.5392400364333209), ('Following Lane at speed 0.73', 0.7272658740247249), ('Following Lane at speed 1.37', 1.3675394461888557), ('Following Lane at speed 0.95', 0.9530879226553614)],
    ['No Data', 'No Data', 'No Data', 'No Data'],
    ['No Data', 'No Data', ('Following Lane at speed 0.50', 0.5035237950175756), ('Following Lane at speed 0.33', 0.33417837334215067)],
    [('Following Lane at speed 1.40', 1.4005948743730938), ('Following Lane at speed 0.52', 0.5238918821398532), 'No Data', 'No Data'],
    ['No Data', 'No Data', 'No Data', 'No Data'],
    [('Following Lane at speed 1.12', 1.1229521963275328), ('Following Lane at speed 0.25', 0.25335426176663894), ('Following Lane at speed 0.10', 0.1016280559874524), ('Following Lane at speed 0.18', 0.1809043134276479)],
    [('Following Lane at speed 1.00', 1.0030487991523873), ('Following Lane at speed 1.17', 1.1716095401496431), ('Following Lane at speed 1.42', 1.415554332286716), ('Following Lane at speed 1.34', 1.3388578132347129)],
    [('Following Lane at speed 4.87', 4.872963828986651), ('Following Lane at speed 5.76', 5.756444490137956), ('Changing to Right Lane to the right at speed 6.45', 6.454973957676724), ('Following Lane at speed 6.70', 6.6976588648205215)],
    [('Following Lane at speed 0.34', 0.34101476365802813), ('Following Lane at speed 0.15', 0.14673612147890222), ('Following Lane at speed 0.15', 0.14594103292096783), 'No Data'],
    [('Following Lane at speed 0.77', 0.7692631012890178), ('Following Lane at speed 0.25', 0.2531356768713868), ('Following Lane at speed 0.65', 0.6549303581166951), ('Following Lane at speed 0.62', 0.6197082102805345)],
    [('Following Lane at speed 0.39', 0.3876073068804195), ('Following Lane at speed 0.10', 0.10085183017516999), ('Following Lane at speed 0.27', 0.27298440663360385), 'No Data'],
    [('Following Lane at speed 0.19', 0.18762254118666385), ('Following Lane at speed 0.30', 0.3047990564638221), 'No Data', 'No Data'],
    ['No Data', ('Following Lane at speed 2.01', 2.014537540873904), ('Following Lane at speed 0.51', 0.5128310462397698), 'No Data'],
    ['No Data', ('Following Lane at speed 0.54', 0.5375949846087912), ('Following Lane at speed 0.19', 0.18859709928147195), ('Following Lane at speed 0.59', 0.5937830817187928)],
    ['No Data', ('Following Lane at speed 0.40', 0.4043635064088685), ('Following Lane at speed 0.33', 0.33272281644510887), ('Following Lane at speed 0.70', 0.697741151865638)],
    ['No Data', 'No Data', ('Following Lane at speed 0.90', 0.8971609065106022), ('Following Lane at speed 1.27', 1.2724727842525387)],
    ['No Data', 'No Data', 'No Data', ('Following Lane at speed 1.52', 1.517156377583627)],
    ['No Data', 'No Data', 'No Data', 'No Data'],
    ['No Data', 'No Data', 'No Data', 'No Data'],
    ['No Data', 'No Data', 'No Data', 'No Data'],
    ['No Data', 'No Data', 'No Data', 'No Data']
]
