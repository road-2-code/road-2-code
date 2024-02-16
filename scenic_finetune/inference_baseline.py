from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
from test_gen import interpret_action_with_speed
import os
import json
import re
def generate_frame_descriptions(data):
    """Generate a text description for each frame based on car positions."""
    descriptions = []
    for i, frame in enumerate(data):

        position_desc = f"new car at position ({frame[0]}, {frame[1]})"
        descriptions.append(position_desc)

    return "\n".join(descriptions)

def transform_json_data_to_string_with_newlines(json_data):
    transformed_data = []
    for item in json_data:
        lines = [line.strip() for line in item['code'].splitlines() if line.strip()]
        concatenated_string = '\n'.join(lines) + '\n '
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


left_lane_x_range = (-4.5, -1.5)
center_lane_x_range = (-1.5, 1.5)
right_lane_x_range = (1.5, 4.5)
time_interval = 0.5  # Each frame is 0.5 seconds

# Map lane ranges for easy reference
lane_ranges = {
    "left": left_lane_x_range,
    "center": center_lane_x_range,
    "right": right_lane_x_range
}

# base_model = "codellama/CodeLlama-7b-hf"
base_model = "meta-llama/Llama-3.1-8B"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(base_model,use_fast=True)
# model = PeftModel.from_pretrained(model, './carla-code-llama-func-instruct/checkpoint-7000')
model = PeftModel.from_pretrained(model, './carla-code-llama-func-instruct-baseline/checkpoint-1500')
model = torch.compile(model)




dataset_dir = "/playpen1/jleung18/scenic_finetune/data/track_output_4"
data_samples = []

# for sub_dir in sorted(os.listdir(dataset_dir)):
sub_dir_path = os.path.join(dataset_dir, '10')
    # if os.path.isdir(sub_dir_path):
input_path = os.path.join(sub_dir_path, "coordinates.json")
output_path = os.path.join(sub_dir_path, "track.json")

with open(input_path, "r") as f:
    perceptions = json.load(f)
    actions = interpret_action_with_speed(perceptions, time_interval)
    with open(output_path, "r") as f:
        scenic_code = json.load(f)
    new_input = extract_coordinates(scenic_code)
    instruction = generate_frame_descriptions(new_input)
    context = """Write a Scenic script to place each car at the specified coordinates from the list. Use the Scenic syntax 'new Car offset by (x, y)' for each car, and assign blueprint 'vehicle.lincoln.mkz_2020' and color 'Color(0.3, 0.3, 0.3)'. Here is a sample program: new Car offset by (-8.228522300720215, 2.675042152404785),\nwith blueprint "vehicle.lincoln.mkz_2020",\nwith color Color(0.3,0.3,0.3)\n. Here are the specific commands and coordinates for each car: """
    instruction = context + instruction

# with open(output_path, "w") as f:
#     json.dump(actions, f, indent=4)
# prompt1 = f"""
# ### Input: <s>[INST]'Write a Scenic script to place each car at the specified coordinates from the list. Use the Scenic syntax \'new Car offset by (x, y)\' for each car, and assign blueprint \'vehicle.lincoln.mkz_2020\' and color \'Color(0.3, 0.3, 0.3)\'. Here is a sample program: new Car offset by (-8.228522300720215, 2.675042152404785),\nwith blueprint "vehicle.lincoln.mkz_2020",\nwith color Color(0.3,0.3,0.3)\n.
# Here are the specific commands and coordinates for each car: new car at position (-8.228522300720215, 2.675042152404785)
# new car at position (-4.916201591491699, -13.940065383911133)
# new car at position (-5.799027442932129, -9.37924575805664)
# new car at position (7.405271530151367, 7.089926719665527)
# new car at position (-4.594717025756836, -21.912078380584717)
# new car at position (-8.273386001586914, 1.0866994857788086)
# new car at position (-7.733608245849609, -2.8272342681884766)
# new car at position (-8.12592887878418, -4.790139198303223)
# new car at position (2.5641260147094727, -12.081862449645996)
# new car at position (5.522063255310059, 18.604622840881348)
# new car at position (-5.425546646118164, -12.537277221679688)
# new car at position (-4.256647109985352, -18.567492485046387)
# new car at position (-9.16549301147461, 18.74779987335205)
# new car at position (-9.6126708984375, 21.750248908996582)
# new car at position (-10.182987213134766, 0.5588865280151367)
# new car at position (inf, inf)
# new car at position (-inf, inf)
# new car at position (-7.618303298950195, -8.913053512573242)
# new car at position (-8.912681579589844, 8.67041301727295)
# new car at position (-4.796184539794922, 33.87990474700928)
# new car at position (-8.781452178955078, 14.484057426452637)
# new car at position (-2.693026542663574, -24.94182300567627)
# new car at position (-8.608835220336914, 7.918825149536133)' [/INST]
# ### Response:
# """


# prompt = """
# ### Input:
# "[INST]Generate a function that does the following:
# car 1 follows lane at target speed 2.04. When within a distance to any objects of 10, follow lane at speed 0.[/INST]
# ### Response:
#
# ### Input:
# "[INST]Generate a function that does the following:
# car 1 follows lane at target speed 2.04. When within a distance to any objects of 10, follow lane at speed 0.[/INST]
# ### Response:
#
# ### Input:
# "[INST]Generate a function that does the following:
# car 1 follows lane at target speed 5.95. When within a distance to any objects of 10, follow lane at speed 0.[/INST]
# ### Response:
# """

prompt = """
### Input:
<s>[INST]Create a Scenic script for placing cars and simulating their behavior, each car must be positioned at specified coordinates using the syntax new Car offset by (x, y), with each assigned the blueprint "vehicle.lincoln.mkz_2020" and color Color(0.3, 0.3, 0.3). Example commands for placement include new Car offset by (2.8, -7.5), with blueprint "vehicle.lincoln.mkz_2020", with color Color(0.3,0.3,0.3). The coordinates for placement are: new car offset by (2.8, -7.5)\nnew car offset by (3.2, -5.4)\nnew car offset by (2.0, -11.4)\nnew car offset by (1.1, -13.0)\nnew car offset by (-0.1, -15.9)\nnew car offset by (-0.9, -12.7)\nnew car offset by (1.9, -10.1)\nnew car offset by (-0.4, 24.0)\nnew car offset by (-2.9, -13.9)\nnew car offset by (-5.9, 26.3)\nnew car offset by (1.0, 20.5)\nnew car offset by (3.0, 21.9) Create the script that defines car behaviors. For instance, a behavior definition might look like: behavior Behavior_car1(): do CustomFollowLaneBehavior(0.80) for 0.5 seconds\n do CustomFollowLaneBehavior(1.92) for 0.5 seconds\n do CustomLaneChangeBehavior(network.laneSectionAt(self)._laneToLeft). Behavior specifications include commands for various cars: Car Car 1:\n  First follows lane at speed 0.80 for 0.5 seconds, then follows lane at speed 1.92 for 0.5 seconds, then follows lane at speed 1.74 for 0.5 seconds, and finally changes lane to the left at speed 1.70 for 0.5 seconds.\nCar 2:\n  First follows lane at speed 1.60 for 0.5 seconds, then follows lane at speed 2.23 for 0.5 seconds, and finally changes lane to the left at speed 1.63 for 0.5 seconds.\nCar 3:\n  First follows lane at speed 0.36 for 0.5 seconds, then follows lane at speed 0.19 for 0.5 seconds, and finally follows lane at speed 0.12 for 0.5 seconds.\nCar 4:\n  First follows lane at speed 0.99 for 0.5 seconds, and finally follows lane at speed 0.17 for 0.5 seconds.\nCar 5:\nCar 6:\nCar 7:\n  First follows lane at speed 1.15 for 0.5 seconds, and finally follows lane at speed 1.78 for 0.5 seconds.\nCar 8:\n  First follows lane at speed 0.56 for 0.5 seconds.\nCar 9:\n  First follows lane at speed 0.47 for 0.5 seconds.\nCar 10:\n  First follows lane at speed 3.71 for 0.5 seconds.\nCar 11:\nCar 12:\n These above should be combined into a cohesive Scenic script. Can you generate a Scenic program for me with these instructions? [/INST]
### Response:
"""

model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=1500)[0],skip_special_tokens=True))
