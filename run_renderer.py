import scenic
# from utils.simulator import CarlaSimulator
import carla

import time
import socket
import os
import argparse

import subprocess

from scenic.simulators.carla.simulator import CarlaSimulator

from utils.config import IMAGE_DIR, TRAINING_DATA_DIR, SEG_DIR, DEPTH_DIR

from scenic.core.errors import InvalidScenarioError

from scenic.syntax.veneer import deactivate

LOG_DIR = 'logs/carla'

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, required=True, default=1)
parser.add_argument('--end', type=int, required=True, default=100)
parser.add_argument('--port', type=int, default=2000)
# Do we render the input files or output files
# use relative path ./test/programs/{scene_i}_output.scenic
parser.add_argument('--render-outputs', action='store_true')
parser.add_argument('--multiview', action='store_true')
# Optional; use if we have scenes directly in the folder
parser.add_argument('--folder', type=str)
parser.add_argument('--folder-2', type=str)
# Only use if --folder exists
parser.add_argument('--folder-out', type=str)

parser.add_argument('--exp', type=str)
parser.add_argument('--test', action='store_true')
parser.add_argument('--config', type=str, default='waymo')

# Render the respective daetaset.
# use if --render-outputs is True
parser.add_argument('--dataset-config', type=str)
# Only use if --render-outputs is False
parser.add_argument('--dataset-dir', type=str, default="data")

args = parser.parse_args()



# transform_root = TRAINING_DATA_DIR

# os.makedirs(IMAGE_DIR, exist_ok=True)
# os.makedirs(DEPTH_DIR, exist_ok=True)
# os.makedirs(transform_root, exist_ok=True)

start_time = time.time()

print("Starting...", flush=True)
# time.sleep(60)

import psutil

dataset_dir = args.dataset_dir


def get_camera_matrix(camera, camera_bp):
   # https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/lidar_to_camera.py
   import numpy as np
   image_w = camera_bp.get_attribute("image_size_x").as_int()
   image_h = camera_bp.get_attribute("image_size_y").as_int()
   fov = camera_bp.get_attribute("fov").as_float()
   focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
   
   K = np.identity(3)
   K[0, 0] = K[1, 1] = focal
   K[0, 2] = image_w / 2.0
   K[1, 2] = image_h / 2.0
   
   world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
   # camera_2_world_ = np.array(camera.get_transform())
   # Premultiply by the "carla -> standard image coords" conversion;
   # Postmultiply by the "scenic -> carla" coordinate conversion
   # print("K", K, "camera 2 world", world_2_camera)
   world_2_camera = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0]]) @ world_2_camera @ np.array([[1,0,0,0],
                                               [0,-1,0,0],
                                               [0,0,1,0],
                                               [0,0,0,1]])

   camera_2_world = np.array(camera.get_transform().get_matrix())
   camera_2_world = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0]]) @ camera_2_world @ np.array([[1,0,0,0],
                                               [0,-1,0,0],
                                               [0,0,1,0],
                                               [0,0,0,1]])
   
   return K @ world_2_camera, world_2_camera
   

def check_if_in_frame(camera_matrix, transforms):
   """ Check if a car is in the camera's frame. """
   import numpy as np
   coordinates = [camera_matrix @ np.array([l[0], l[1], l[2], 1]) for l, _o in transforms]
   # print("non-normalised coords", coordinates)
   coordinates = [np.array([p[0] / p[2], p[1] / p[2]]) for p in coordinates if p[2] > 0]
   # print("coords", coordinates)
   for c in coordinates:
      if (0 <= c[0] <= 960 and 0 <= c[1] <= 540):
         return True
   return False


def render_atpoint():
   pass

def generate(scene_i: int, dir, args):
   """
   Render a single scene in Carla given a program
   
   scene_i: scene identifier.
   
   """
   print("Port", args.port)
   time_start = time.time()
   should_render_outputs = args.render_outputs
   if should_render_outputs:
      if args.dataset_config is not None:
         path = f'{dir}/{args.exp}/test/programs/{scene_i}.scenic'
      elif args.folder is not None or args.folder_2 is not None:
         import glob
         import os

         # Set the directory you want to search in
         # Get a list of all matching files
         path = f'{dir}/{scene_i}*.scenic'
         files = glob.glob(path)
         path = files[0]
      elif args.test:
         path = f'{dir}/{args.exp}/test/programs/{scene_i}.scenic'
      else:
         path = f'{dir}/{args.exp}/train/programs/{scene_i}.scenic'
      if not os.path.exists(path):
         print(f"{path} does not exist")
         return
      
      print("Loading from", path)

      CarlaSimulator.scene_i = scene_i
      CarlaSimulator.args = args
      CarlaSimulator.dir = dir # output dir
      CarlaSimulator.multiview = args.multiview
      scenario = scenic.scenarioFromFile(path,
                                       model='scenic.simulators.carla.model',params={'timeout':120, 'render': False, 'port':args.port, 'timestep':0.1})

   else:
      CarlaSimulator.scene_i = scene_i
      CarlaSimulator.args = args
      CarlaSimulator.dir = dir
      CarlaSimulator.multiview = args.multiview
      scenario = scenic.scenarioFromFile(f'{dataset_dir}/scenic/{scene_i}.scenic',
                                       model='scenic.simulators.carla.model',params={'timeout':120, 'render': False, 'port':args.port, 'timestep':0.1, 'scene_i': f"{scene_i}"})
      print(f'loaded {dataset_dir}/scenic/{scene_i}.scenic')
   scene, _ = scenario.generate(maxIterations=20)
   simulator = scenario.getSimulator()
   # simulator = CarlaSimulator()
   
   #settings = simulator.world.get_settings()
   #settings.fixed_delta_seconds = 0.1
   #settings.synchronous_mode = True

   #simulator.world.apply_settings(settings)
   time_end = time.time()

   print("created simulator")
   # Refer to CarlaSimulator code in Scenic folder for full implementation.
   simulation = simulator.simulate(scene, maxSteps=90)
   print("running simulator")

   locations = []
   if simulation:  # `simulate` can return None if simulation fails
         result = simulation.result
         for i, state in enumerate(result.trajectory):
               egoPos = state[0]
               locations.append(state)
#   
   simulator.destroy()
   del simulator.world
   del simulator.client
   
   
import tqdm, os


import yaml

dataset_config = {}
if args.dataset_config is not None:
   with open(args.dataset_config) as stream:
      try:
         dataset_config = yaml.safe_load(stream)
      except yaml.YAMLError as exc:
         print(exc)
         raise exc

import multiprocessing

# Define a function that raises an exception
def render(i):
   try:
      print(f"scene {i}")
      if args.render_outputs:
         # if args.dataset_config is not None:
         #    dataset_folder = dataset_config['folder']
         #    
         #    path = f'{dataset_folder}/logs/{args.exp}/test/programs/{i}_output.scenic'
         #    
         #    args.test = True # use testing folder to out put results
         # elif args.test:
         #    path = f'{LOG_DIR}/{args.exp}/test/programs/{i}_output.scenic'
         # if os.path.exists(path):
            if args.dataset_config is not None:
               # KITTTI
               dataset_folder = dataset_config['folder']
               for j in tqdm.tqdm(range(0,400)):
                  generate(f"{i}_{j}_output", f"logs/kitti", args)
            elif args.folder is not None:
              # if render things from a folder
               dataset_folder = args.folder
               # os.makedirs(args.folder_out, exist_ok=True)
               generate(f"{i}", args.folder, args)
            elif args.folder_2 is not None:
              # if render things from a folder
               dataset_folder = args.folder_2
               # os.makedirs(args.folder_out, exist_ok=True)
               generate(f"{i}", args.folder_2, args)
            else:
               for j in tqdm.tqdm(range(0,400)):
                  generate(f"{i}_{j}", f"{LOG_DIR}", args)
               #generate(f"{i}_0", LOG_DIR)
      elif not args.render_outputs and os.path.exists(f'{dataset_dir}/scenic/{i}.scenic'):
         generate(i, f"{dataset_dir}", args) # generation of Scenic data
      else:
         print("scene not found")
   except InvalidScenarioError as e:
      import traceback
      print(f">>> UNABLE TO RENDER scene {i}:")
      print(traceback.format_exc())
      # manually deactivate Scenic
   except Exception as e:
      import traceback
      print(f">>> UNABLE TO RENDER scene {i}:")
      print(traceback.format_exc())


for i in tqdm.tqdm(range(args.start,args.end)):
   # path = f'{LOG_DIR}/{args.exp}/train/programs/{i}_output.scenic'
   import time
   
   # Create and start the thread, passing the queue as an argument
   q = multiprocessing.Queue()

    # Create a new process
   p = multiprocessing.Process(target=render, args=(i,))
   p.start()

   # Wait for the thread to finish
   p.join()




