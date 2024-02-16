import re
import json
import os

def extract_offsets(json_data):
    """
    Extracts scenic offsets from the JSON data, based on "new Car offset by" values.

    Parameters:
    - json_data: List of dictionaries containing the 'code' key with offset coordinates as strings.

    Returns:
    - List of tuples with the extracted offset coordinates.
    """
    offsets = []

    for item in json_data:
        code = item['code']
        # Find the coordinates inside the first "new Car offset by (x, y)"
        match = re.search(r'offset by \(([-\d.einf]+),\s*([-\d.einf]+)\)', code)
        if match:
            x, y = match.groups()
            x = float(x) if x != 'inf' and x != '-inf' else float(x)
            y = float(y) if y != 'inf' and y != '-inf' else float(y)
            offsets.append((x, y))

    return offsets


def preprocess_data(data, offsets):
    """
    Preprocesses the input data by replacing only the first two `None` values in each list
    with the corresponding offset from the extracted offsets list.

    Parameters:
    - data: List of lists with coordinate pairs or None values.
    - offsets: List of tuples containing default coordinate pairs to replace the first two None values.

    Returns:
    - List of lists with the first two `None` values replaced by offsets.
    """
    offset_index = 0  # Keep track of which offset to use

    for instance in data:
        # Track how many `None`s have been replaced
        none_replacements = 0

        for i in range(len(instance)):
            # Replace only the first two `None`s with offset values
            if instance[i] is None and offset_index < len(offsets):
                if none_replacements == 0:
                    # Replace the first None with the x coordinate
                    instance[i] = offsets[offset_index][0]
                    none_replacements += 1
                elif none_replacements == 1:
                    # Replace the second None with the y coordinate
                    instance[i] = offsets[offset_index][1]
                    offset_index += 1  # Move to the next offset only after replacing both None values
                    none_replacements += 1
                    break  # Stop after replacing the first two None values

    return data

# Example usage
json_data = [
    {
        "code": "new Car offset by (-8.228522300720215, 2.675042152404785),\n   with blueprint \"vehicle.lincoln.mkz_2020\",\n   with color Color(0.3,0.3,0.3)"},
    {
        "code": "new Car offset by (-4.916201591491699, -13.940065383911133),\n   with blueprint \"vehicle.lincoln.mkz_2020\",\n   with color Color(0.3,0.3,0.3)"},
    # Add more items as needed
]

data = [
    [(None, None), [1, 2], None, None],  # Sample data
    [(None, None), [3, 4], None, [5, 6]],
    # Add more instances as needed
]

# Extract offsets from JSON data
offsets = extract_offsets(json_data)

# Preprocess data
processed_data = preprocess_data(data, offsets)
print(processed_data)
dataset_dir = "./data/track_output_3"


def transform_json_data_to_string_with_newlines(json_data):
    """
    Transforms JSON data by removing the "code" key and converting the content of each "code" entry
    into a single formatted string with '\n' preserved between lines.

    Parameters:
    - json_data: List of dictionaries containing the 'code' field.

    Returns:
    - List of strings, where each string contains the concatenated lines of a "code" entry with '\n' preserved.
    """
    transformed_data = []

    for item in json_data:
        # Split the 'code' content into lines and strip whitespace, adding '\n' to preserve line breaks
        lines = [line.strip() for line in item['code'].splitlines() if line.strip()]
        # Join the lines into a single string with '\n'
        concatenated_string = '\n '.join(lines) + '\n '
        transformed_data.append(concatenated_string)

    return transformed_data

def extract_coordinates(json_data):
    coordinates = []

    for item in json_data:
        code = item['code']
        # Use regex to extract the coordinates within "offset by (...)"
        match = re.search(r'offset by \(([-\d.einf]+),\s*([-\d.einf]+)\)', code)
        if match:
            x, y = match.groups()
            x = float(x) if x != 'inf' and x != '-inf' else float(x)
            y = float(y) if y != 'inf' and y != '-inf' else float(y)
            coordinates.append([x, y])

    return coordinates
for sub_dir in sorted(os.listdir(dataset_dir)):
    sub_dir_path = os.path.join(dataset_dir, sub_dir)
    if os.path.isdir(sub_dir_path):
        input_path = os.path.join(sub_dir_path, "coordinates.json")
        output_path = os.path.join(sub_dir_path, "track.json")

        with open(input_path, "r") as f:
            perceptions = json.load(f)
        with open(output_path, "r") as f:
            scenic_code = json.load(f)

        new_input = extract_coordinates(scenic_code)
        new_output = transform_json_data_to_string_with_newlines(scenic_code)


        # instruction = generate_frame_descriptions(perceptions)