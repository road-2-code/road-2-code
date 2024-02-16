import torch
from datasets import load_dataset
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     TrainingArguments,
# DataCollatorForSeq2Seq,
#     pipeline,
#     logging,
# )
import re
import os
from trl import SFTConfig, SFTTrainer
from peft import (
    LoraConfig, get_peft_model,get_peft_model_state_dict,prepare_model_for_kbit_training,set_peft_model_state_dict)
from datasets import Dataset
import json,sys
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq,QuantoConfig
from test_gen import interpret_action_with_speed
from test_code_gen import generate_behavior_code, print_car_behavior
# base_model = "codellama/CodeLlama-70b-hf"
torch.cuda.empty_cache()

# base_model = "codellama/CodeLlama-7b-hf"
base_model = "meta-llama/Llama-3.1-8B"

quantization_config = QuantoConfig(weights="int8")

input = [[[-2.908048917380029, 21.198068111708153], [-2.7997535576412247, 21.41127550869669], [-2.785522380309658, 21.579753175129895], [-2.893904003624584, 21.911104113346994], [-2.8712207663327263, 22.474721930582632]], [[-2.625999065941187, -14.595141357233615], [-2.7036110481847118, -12.500402227241949], [-2.670321868180281, -10.495119584102326], [-2.634539977538907, -7.95963574411013], [-2.6694143466741025, -5.569380926538372]], [[0.20219095958645994, -20.370380874926923], [0.05497713352315259, -20.90273423912845], [0.10333128654201573, -21.28781246043951], [0.2766521235996606, -21.395482052972113], [0.25313286485516073, -20.889889116041275]], [[0.41091909787041914, -31.85944244227727], [0.2548108332393895, -33.10347382822394], [0.3407588054558345, -34.415803450109166], [0.510454243851882, -35.275300497470425], [0.41006177427880175, -35.61184696699689]]]


def convert_to_code_llama_format(input_path, output_path):
    with open(input_path, "r") as f:
        perceptions = json.load(f)
    with open(output_path, "r") as f:
        scenic_code = f.read()
    perceptions_str = json.dumps(perceptions, indent=2)
    instruction = f"""
`Generate Scenic code from the following perception`s:
{perceptions_str} """
    response = f"""{scenic_code}"""
    return instruction,response


def generate_behavior_script(data):
    behavior_scripts = []

    for car_index, car_data in enumerate(data, start=1):
        behavior_script = f"behavior Behavior_car{car_index}():\n"
        has_valid_data = False

        # Iterate through the car's data in pairs of 2 (representing 0.5 seconds)
        for i in range(0, len(car_data), 2):
            # Check if both entries in the 0.5-second segment are valid
            if car_data[i] != 'No Data' and i + 1 < len(car_data) and car_data[i + 1] != 'No Data':
                # Combine the speeds and descriptions for 0.5 seconds
                first_entry, first_speed = car_data[i]
                second_entry, second_speed = car_data[i + 1]

                # Average the speed for a combined 0.5-second behavior
                avg_speed = (first_speed + second_speed) / 2
                behavior_script += f"    do CustomFollowLaneBehavior({avg_speed:.2f}) for 0.5 seconds\n"
                has_valid_data = True

        if has_valid_data:
            behavior_scripts.append(behavior_script)

    return behavior_scripts


def generate_behavior_descriptions(data):
    behavior_descriptions = []

    for car_index, car_data in enumerate(data, start=1):
        description = f"Car {car_index}:\n"
        has_valid_data = False

        # Iterate through the car's data in pairs of 2 (representing 0.5 seconds)
        for i in range(0, len(car_data), 2):
            # Check if both entries in the 0.5-second segment are valid
            if car_data[i] != 'No Data' and i + 1 < len(car_data) and car_data[i + 1] != 'No Data':
                # Combine the speeds for 0.5 seconds
                first_entry, first_speed = car_data[i]
                second_entry, second_speed = car_data[i + 1]

                # Average the speed for a combined 0.5-second behavior
                avg_speed = (first_speed + second_speed) / 2
                description += f"  Follow lane at speed {avg_speed:.2f} for 0.5 seconds\n"
                has_valid_data = True

        if has_valid_data:
            behavior_descriptions.append(description)

    return behavior_descriptions



def generate_frame_descriptions(data):
    """Generate a text description for each frame based on car positions."""
    descriptions = []
    for i, frame in enumerate(data):

        position_desc = f"new car offset by ({frame[0]}, {frame[1]})"
        descriptions.append(position_desc)

    return "\n".join(descriptions)

def truncate_text(text, n):
    """Truncate text up to the nth newline."""
    # Split text by newline and join back up to the nth occurrence
    truncated_text = '\n'.join(text.splitlines()[:n])
    return truncated_text

def generate_tracking_context(data):
    context = "# Video Tracking Scenario Overview\n"
    context += "In this scenario, we track multiple cars over a series of frames. The car at `[0, 0]` represents our car, while other coordinates indicate the positions of surrounding cars. Each frame provides a snapshot of the cars' relative movements and lane assignments over time.\n\n"
    context += "# Frame-by-Frame Tracking\n"

    for frame_idx, frame in enumerate(data, start=1):
        context += f"\n## Frame {frame_idx}\n"
        context += "In this frame, the main car (our car) is at position `[0, 0]`. The following cars are positioned as described:\n"

        main_car_position = frame["transforms"][0]
        other_cars = frame["transforms"][1:]  # Exclude the main car's position for descriptions
        lanes = frame["lanes"]

        for car_idx, (position, lane) in enumerate(zip(other_cars, lanes[1:]), start=1):
            x, y = position
            lane_desc = f"in lane {lane}" if lane >= 0 else "off the road"
            distance = ((x - main_car_position[0]) ** 2 + (y - main_car_position[1]) ** 2) ** 0.5
            direction = "ahead" if y < 0 else "behind" if y > 0 else "alongside"
            context += f"- **Car {car_idx}**: Located at (`x = {x:.3f}`, `y = {y:.3f}`), approximately {distance:.2f} units {direction} our car, {lane_desc}.\n"
        # Describe car behavior if present
        # desc = frame.get("desc", "No specific behavior description provided.")
        # context += f"**Behavior**: {desc}\n"
    context += "\n# Scenic Code Generation\n"
    context += "The Scenic code below will reflect these frame-by-frame tracking details, with each car's relative position and behavior accurately described for simulation.\n"
    return context

dataset_dir = "./data/track_output"
data_samples = []

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


def extract_behavior(data):
    behavior_descriptions = ""

    for car_index, car_data in enumerate(data, start=1):
        behavior_descriptions += f"Car {car_index}:\n"
        for i in range(0, len(car_data), 2):  # Iterate in pairs for 0.5-second intervals
            if car_data[i] != 'No Data' and i + 1 < len(car_data) and car_data[i + 1] != 'No Data':
                behavior_descriptions += f"  {car_data[i][0]} for 1.0 seconds\n"
                behavior_descriptions += f"  {car_data[i + 1][0]} for 1.0 seconds\n"
            elif car_data[i] != 'No Data':
                behavior_descriptions += f"  {car_data[i][0]} for 1.0 seconds\n"
            elif i + 1 < len(car_data) and car_data[i + 1] != 'No Data':
                behavior_descriptions += f"  {car_data[i + 1][0]} for 1.0 seconds\n"

    return behavior_descriptions.strip()

for sub_dir in sorted(os.listdir(dataset_dir)):
    sub_dir_path = os.path.join(dataset_dir, sub_dir)
    if os.path.isdir(sub_dir_path):
        input_path = os.path.join(sub_dir_path, "coordinates.json")
        output_path = os.path.join(sub_dir_path, "track.json")

        with open(input_path, "r") as f:
            perceptions = json.load(f)
            actions = interpret_action_with_speed(perceptions, 1) # 0.5)
            function_behavior = generate_behavior_code(actions)
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


        with open(output_path, "r") as f:
            scenic_code = json.load(f)
            for car in scenic_code:
                car['code'] = re.sub(
                    r'(-?\d+\.\d+)',
                    lambda x: f"{float(x.group()):.1f}",
                    car['code']
                )
        new_input = extract_coordinates(scenic_code)
        new_output = transform_json_data_to_string_with_newlines(scenic_code)
        out = ""
        for i in new_output:
            out +=i

        car_blocks = re.split(r'(?=new Car offset by)', out)

        # Generate labeled car output
        labeled_output = []
        for index, block in enumerate(car_blocks):
            if block.strip():  # Ensure it's not an empty block
                if index == 0:
                    labeled_output.append(f"ego = {block.strip()}")
                else:
                    labeled_output.append(f"car{index} = {block.strip()}")
        placement = "\n\n".join(labeled_output)
        placement += "\n "

        steps = "Let’s think step by step. "
        new_output = steps + "Step 1: Each car must be placed accurately at the specified (x, y) coordinates to set up the initial state of the simulation. The positions represent the starting point of each car in the scene. Places the cars at their specified coordinates using the syntax:" + "\n" + placement
        func_output =  "Step 2: The function descriptions provided for each car outline how they should behave over time. This step is crucial for simulating movement and interactions over multiple frames. Behavior function syntax is:" + "\n" +function_behavior
        final_output = "Step 3: Integrating the car placements and behavior functions into a single Scenic program ensures that the simulation runs cohesively. This step combines the placement setup from Step 1 with the function behaviors from Step 2.\n"
        final  = "Therefore, the Scenic script is --> " +placement + "\n" + function_behavior
        output = new_output + "\n" + func_output + "\n" + final_output + final
        try:
            instruction = generate_frame_descriptions(new_input)
            context = """To create a Scenic script for placing cars and simulating their behavior, each car must be positioned at specified coordinates using the syntax new Car offset by (x, y), with each assigned the blueprint "vehicle.lincoln.mkz_2020" and color Color(0.3, 0.3, 0.3). Example commands for placement include new Car offset by (2.8, -7.5), with blueprint "vehicle.lincoln.mkz_2020", with color Color(0.3,0.3,0.3). The coordinates for placement are: \n"""
            instruction = context + instruction
            step2 = instruction + " \nThe script defines behavior functions for cars over time with each frame lasting 1.0 seconds. Skip for cars with empty behaviors. For instance, a behavior definition might look like: behavior Behavior_car1(): do CustomFollowLaneBehavior(0.80) for 1.0 seconds\n do CustomFollowLaneBehavior(1.92) for 1.0 seconds\n do CustomLaneChangeBehavior(network.laneSectionAt(self)._laneToLeft). Behavior specifications include commands for various cars: \n"
            step2 = step2 + formatted_func
            step3 = step2 + " Finally, these placement and behavior definitions should be combined into a cohesive Scenic script to simulate the scene effectively."
            data_samples.append({"instruction": step3, "response": output})

        except Exception as e:
            print(f"Failed to process {sub_dir}: {e}")

dataset = Dataset.from_list(data_samples,split="train")
train_dataset = dataset.train_test_split(test_size=0.1)["train"]
eval_dataset = dataset.train_test_split(test_size=0.1)["test"]

compute_dtype = getattr(torch, "float16")

access_token ='' # ADD YOUR TOKEN
# access_token = 'access'
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
    use_auth_token=access_token
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model,use_auth_token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=2048,
        padding=False,
        return_tensors=None,
    )
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""

### Input:
<s>[INST]{data_point["instruction"].strip()} [/INST]

### Response:
{data_point["response"] + " END"}
"""
    # tokenize(full_prompt)['labels'] *= -100
    return tokenize(full_prompt)


model.train() # put model back into training mode
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
tokenized_train_dataset = dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

# print("insdt >>>", dataset[0]['instruction'])
# print("resp >>>", dataset[0]['response'])
# raise ""


batch_size = 1
per_device_train_batch_size = 1
gradient_accumulation_steps = batch_size // per_device_train_batch_size
output_dir = "carla-code-llama-func-instruct-cot8"
import datetime
training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        max_steps=1500,
        learning_rate=3e-4,
        fp16=True,
        # logging_steps=10,
        optim="adamw_torch",
        # evaluation_strategy="steps", # if val_set_size > 0 else "no",
        save_strategy="steps",
    # eval_steps=100,
        save_steps=100,
        output_dir=output_dir,
        load_best_model_at_end=False,
        group_by_length=True, # group sequences of roughly the same length together to speed up training
        run_name=f"codellama-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}", # if use_wandb else None,
    )

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)
if torch.__version__ >= "2" and sys.platform != "win32":
    print("compiling the model")
    model = torch.compile(model)
    torch.cuda.empty_cache()

trainer.train()
torch.cuda.empty_cache()

