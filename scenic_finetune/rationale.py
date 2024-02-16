


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# Replace 'llama-3.1' with the appropriate model name or path
from utils.behavior import *
model_name = "meta-llama/Llama-3.1-8B"

coordinates = [[[-14.015625, 0.007444620132446289], [-15.9613037109375, 0.008418083190917969], [-17.684356689453125, 0.009340047836303711], [-19.452606201171875, 0.010279655456542969], [-21.21466064453125, 0.011215448379516602]]]

# Extract the first element of each sublist and format the output
instruction = ""
for sublist in coordinates:
    x, y = sublist[0]
    instruction += f"\nnew car offset by ({x:.1f}, {y:.1f})"

actions = interpret_action_with_speed(coordinates,0.5)
func = extract_behavior(actions)
formatted_func = ""
for car_block in func.strip().split("\nCar "):
    if car_block.strip():
        car_lines = car_block.strip().split("\n")
        car_number = car_lines[0].strip(":")
        behaviors = car_lines[1:]

        if not behaviors:
            formatted_func += f"Car {car_number}:\n"
        else:
            formatted_behaviors = []
            for i, behavior in enumerate(behaviors):
                behavior = behavior.strip()
                if i == 0:
                    formatted_behaviors.append(f"First {behavior.lower()}")
                elif i == len(behaviors) - 1:
                    formatted_behaviors.append(f"and finally {behavior.lower()}")
                else:
                    formatted_behaviors.append(f"then {behavior.lower()}")

            behavior_description = ", ".join(formatted_behaviors)
            formatted_func += f"Car {car_number}:\n  {behavior_description}.\n"

context = """To create a Scenic script for placing cars and simulating their behavior, each car must be positioned at specified coordinates using the syntax new Car offset by (x, y), with each assigned the blueprint "vehicle.lincoln.mkz_2020" and color Color(0.3, 0.3, 0.3). Example commands for placement include new Car offset by (2.8, -7.5), with blueprint "vehicle.lincoln.mkz_2020", with color Color(0.3,0.3,0.3). The coordinates for placement are: """
instruction = context + instruction
step2 = instruction + " Generate behavior functions for cars over time, where each frame lasts 0.5 seconds. For cars with no defined behaviors, skip them. For each car, specify its behaviors across frames as follows: use do CustomFollowLaneBehavior(<speed>) for 0.5 seconds for any lane-following behavior, where <speed> is the given speed value, and do CustomLaneChangeBehavior(network.laneSectionAt(self).<direction>) for lane-changing behavior, where <direction> is the specified lane direction. Follow these patterns to define each car's behavior sequence accurately. Behavior specifications include commands for various cars: "
step2 = step2 + formatted_func
step3 = step2 + " Finally, these placement and behavior definitions should be combined into a cohesive Scenic script to simulate the scene effectively. Can you generate a Scenic program for me with these instructions? Focus on reproducing the output code verbatim, not just replicating its behavior."

