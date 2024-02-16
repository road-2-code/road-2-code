from pandaset import DataSet
import numpy as np

# Load dataset
dataset = DataSet('/work/users/j/l/jleung18/pandaset')

for i in range(1,50):
    try:
        # Example: Load a specific sequence
        sequence = dataset[f"{i:03d}"]  # Replace '00' with the actual sequence you want

        sequence = sequence.load_cuboids()
        # Load object annotations for this sequence
        # sequence = sequence.load_lidar().load_cuboids()

        print(len(sequence.cuboids.data))
    except KeyError:
        continue
    # Access coordinates of bounding boxes for tracked objects
    # for obj in sequence.cuboids.data[0]:
        # print(f"Object ID: {obj['track_id']}")
        # print(f"Bounding box coordinates: {obj['bbox']}")
