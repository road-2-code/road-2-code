import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
import numpy as np

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

def nuscenes():
    # Path to the nuScenes dataset
    nusc = NuScenes(version='v1.0-trainval', dataroot='/work/users/j/l/jleung18/carla/data/nuscenes_data',verbose=True)
    #nusc = NuScenes(version='v1.0-trainval', dataroot='/work/users/j/l/jleung18/carla/UniAD/data/nuscenes_2',verbose=True)

    selected = [("967082162553397800", 5, 100, 200),
                ]

    chosen_idx = None
    # if f_name == "segment-141184560845819621_10582_560_10602_560_with_camera_labels.tfrecord":
        # breakpoint()


    # Choose a scene from the dataset
    scenes = nusc.scene
    scene_tokens = []
    scenes_frames = []
    print("N scenes", len(scenes))
    for scene in scenes[0:]:
        # Get the first sample token of the chosen scene
        scene_token = scene['token']
        scene_tokens.append(scene_token)
        first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
        
        f_name = scene_token
        
        if f_name not in [s[0] for s in selected]:
            pass
        # continue
        else:
            chosen_idx = next((x for x in selected if x[0] == f_name), None)[1]//5
            width_bound = next((x for x in selected if x[0] == f_name), None)[2]
            length_bound = next((x for x in selected if x[0] == f_name), None)[3]

        # Collect 5 consecutive frames (samples)
        frame_tokens = []
        current_sample_token = first_sample_token
        
        frames = []
        
        chosen_idx = None
        width_bound = 15
        height_bound = 100
        # if f_name == "segment-17065833287841703_2980_000_3000_000_with_camera_labels.tfrecord":
        #     breakpoint()
        if f_name not in [s[0] for s in selected]:
            pass
            #dcontinue
        else:
            chosen_idx = next((x for x in selected if x[0] == f_name), None)[1]
            width_bound = next((x for x in selected if x[0] == f_name), None)[2]
            length_bound = next((x for x in selected if x[0] == f_name), None)[3]

        for i in range(30):
            print(i, current_sample_token)
            if current_sample_token == '':
                break
            # Get the sample data for the current frame
            sample = nusc.get('sample', current_sample_token)

            # Append the current frame token to the list
            frame_tokens.append(current_sample_token)

            # Get the token for the next frame (if exists)
            current_sample_token = sample['next']
        for token in frame_tokens:
            frame = nusc.get('sample', token) #['LIDAR_TOP'])  # Example for LIDAR_TOP
    
            frames.append(frame)
        scenes_frames.append(frames)
        exp = "unidepth-0-001-percept"
        # for i in range(1): # f_name in (pbar := tqdm.tqdm(list_dir, total=len(list_dir)) ):
    
    # itertate all scenes
    for scene_fragment_i, frames in enumerate(scenes_frames):
        scene_token = scene_tokens[scene_fragment_i]
        len_dataset = len(frames)
        # if scene_token not in ["3ada261efee347cba2e7557794f1aec8"]: #["3a1850241080418b88dcee97c7d17ed7", "0e37d4a357db4246a908cfd97d17efc6", "87a71839e68b46da8e91fb5f21b50c1c", "798e8504b4364d378270333a349ef508"]:
        if scene_token not in ["406a61a4d394432e95a7e8426a97551f"]: #, "798e8504b4364d378270333a349ef508", "b4b82c4d338a4b6d86835388ce076345"]: #["3a1850241080418b88dcee97c7d17ed7", "0e37d4a357db4246a908cfd97d17efc6", "87a71839e68b46da8e91fb5f21b50c1c", "798e8504b4364d378270333a349ef508"]:
            continue
        # print(len(scenes_frames[0]))
        for chunk_start in tqdm.tqdm(range(0,len_dataset,5)):
            window = frames[chunk_start:chunk_start+5]    
            tracking_centers = {}
            track_ids_map = {}
            j = 0
        
            # this X definition is based on othrerd code
            X = { "name": chunk_start // 5 }
        
            os.makedirs(f"logs/nuscenes/{exp}/track_output__rerun_images/{scene_token}/{X['name']}", exist_ok=True)
            os.makedirs(f"logs/nuscenes/{exp}/track_output__rerun/{scene_token}/{X['name']}", exist_ok=True)
            for frame_i, (frame) in (enumerate( window )):
                print(frame, "frame")
                # Iterate over all annotations in this sample (bounding boxes)
                for annotation_token in frame['anns']:
                        
                    sample_data_token = frame['data']['LIDAR_TOP']
                    sample_data = nusc.get('sample_data', sample_data_token)
                        
                    # Get the ego pose for the sample
                    ego_pose_token = sample_data['ego_pose_token']
                    ego_pose = nusc.get('ego_pose', ego_pose_token)
                    calib_data = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
                        
                    # Ego position (translation in x, y, z)
                    ego_translation = ego_pose['translation']
                    print ("LIDAR pose", ego_pose, "/ token", ego_pose_token)
                    tx, ty, tz = ego_translation
                    # Get the object annotation
                    annotation = nusc.get('sample_annotation', annotation_token)
                        
                    ego_rotation = Quaternion(ego_pose['rotation'])  # (w, x, y, z) quaternion to rotate

                    # Extract object information
                    instance_token = annotation['instance_token']
                    track_id = instance_token
                    category_name = annotation['category_name']
                    box = annotation['translation']  # Center of the bounding box (x, y, z)
                    bbox_size = annotation['size']  # Size of the bounding box (length, width, height)
                    rotation = annotation['rotation']  # Rotation of the bounding box (quaternion)
                    # ======
                    sample_data_token = frame['data']['CAM_FRONT']
                    sample_data = nusc.get('sample_data', sample_data_token)
                    
                    calib_data = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
                    print("Sensor data", calib_data)
                    # Camera calibration (rotation and translation) in ego frame
                    cam_intrinsic = np.array(calib_data['camera_intrinsic'])  # (x, y, z) of camera in ego frame
                    box = np.array(calib_data['translation'])  # (x, y, z) of camera in ego frame
                    # cam_rotation = Quaternion(calib_data['rotation'])  # (w, x, y, z) quaternion for camera
                    print ("cam intrinsic", cam_intrinsic)
                    print ("cam transl", box)
                        
                    with open(f"logs/nuscenes/{exp}/track_output__rerun/{scene_token}/{X['name']}/{frame_i}.txt", "w") as f:
                        f.write("Intrinsic:")
                        f.write(str(cam_intrinsic))
                        f.write("Cam transl:")
                        f.write(str(box))
                        

                    if category_name.startswith("vehicle"): # open_dataset.Label.Type.TYPE_VEHICLE:
                        # Center of the vehicle's bounding box
                            
                        obj_translation = np.array(annotation['translation'])  # (x, y, z)
                        obj_rotation = Quaternion(annotation['rotation'])  # Quaternion for orientation

                        # Transform the object’s global coordinates into the ego vehicle frame
                        obj_in_ego = obj_translation - ego_translation
                        obj_in_ego = ego_rotation.inverse.rotate(obj_in_ego)
            
                        # Transform from ego frame to camera frame
                        # obj_in_cam = cam_rotation.inverse.rotate(obj_in_ego - cam_translation)
                        box = obj_in_ego
                            
                        center_x, center_y, center_z = box[0], box[1], box[2]
                        # note it is reversed
                        center = np.array([center_y, center_x])
                            
                        ### insert into frsme ###
                        if track_id not in track_ids_map:
                            track_ids_map[track_id] = j
                            tracking_centers[j] = [None] * 5
                            tracking_centers[j][frame_i] = center
                                
                            j += 1
                        else:
                            k = track_ids_map[track_id]
                            tracking_centers[k][frame_i] = center
                                
                for index, image in enumerate([frame['data']['CAM_FRONT']]):
                    print(f"Getting image {index} into {scene_token}")
                    camera_data = nusc.get('sample_data', image)
                    image_path = os.path.join(nusc.dataroot, camera_data['filename'])

                    from PIL import Image

                    # Load and display the image using PIL and Matplotlib
                    print(chunk_start, image_path)
                        
                    image = Image.open(image_path)
                    # print(f"Decoding image {index}")
                    # image = tf.image.decode_jpeg(image.image)
                    image.save(f"logs/nuscenes/{exp}/track_output__rerun/{scene_token}/{X['name']}/{frame_i}_{index}.png")
                    # break
        
                if scene_token != "406a61a4d394432e95a7e8426a97551f":
                    break
                for index, image in enumerate([frame['data']['CAM_FRONT']]):
                    token = image
                    camera_data = nusc.get('sample_data', image)
                    idx = 0
                
                    while True:
                        idx += 1
                        if idx > 12 * 2.5:
                            break
                        image_path = os.path.join(nusc.dataroot, camera_data['filename'])

                        from PIL import Image

                        # Load and display the image using PIL and Matplotlib
                        print(chunk_start, image_path)
                            
                        image = Image.open(image_path)
                        # print(f"Decoding image {index}")
                        # image = tf.image.decode_jpeg(image.image)
                        image.save(f"logs/nuscenes/{exp}/track_output__rerun/{scene_token}/{X['name']}/fine_{idx}.png")

                        token = camera_data['next']
                        camera_data = nusc.get('sample_data', token)
                    raise ""
            tracking_meta = {}
            # Scale tracking centers.
            for i in tracking_centers.keys():
                for c in range(len(tracking_centers[i])):
                    if tracking_centers[i][c] is not None:
                        tracking_centers[i][c] *= 0.75
                        tracking_centers[i][c] = [tracking_centers[i][c][0], tracking_centers[i][c][1]]
            for track_id in tracking_centers.keys():
                    
                tracks = tracking_centers[track_id]
                    
                outer = False
                for j, track in enumerate(tracks):
                    # ignore cars that are obviously out of bounds
                    if track is not None:
                        if abs(track[0]) > width_bound * 0.75 or abs(track[1]) > height_bound * 0.75:
                            outer = True
                            break
                if outer:
                    continue
                    
                # Find the first non-None element
                first_element = None
                first_i = 0
                for i, x in enumerate(tracks):
                    if x is not None:
                        first_element = x
                        first_i = i
                        break
                        
                # Find the last non-None element
                last_element = None
                last_i = 1
                for j, x in (reversed(list(enumerate(tracks)))):
                    if x is not None:
                        last_element = x
                        last_i = j
                        break
                delta = (last_i - first_i + 1) * 0.5
                if first_element is None or last_element is None:
                    continue
                # 1 . computeavg velocity. assume 0.5 sec / frame
                velocity =  np.array([(last_element[0] - first_element[0]) / delta,
                            (last_element[1] - first_element[1]) / delta])
                                    
                speed = velocity[1] #(velocity[0]**2 + velocity[1]**2)**0.5
                    
                if delta == 0:
                    speed = 0
                """
                if last_i == first_i:
                    velocity1 =  np.array([0,0])
                    velocity2 =  np.array([0,0])
                else:
                    print(first_i, last_i)
                    velocity1 =  np.array([tracks[first_i+1][0] - (tracks[first_i][0]) / 0.5,
                                            tracks[first_i+1][1] - (tracks[first_i][1]) / 0.5])
                    velocity2 =  np.array([tracks[last_i][0] - (tracks[last_i-1][0]) / 0.5,
                                            tracks[last_i][1] - (tracks[last_i-1][1]) / 0.5])
                    
                for i in range(0, first_i):
                        tracking_centers[track_id][i] = tracking_centers[track_id][first_i] - ((first_i - i) * velocity1)
                        tracking_centers[track_id][i] = tracking_centers[track_id][i].tolist() # - ((first_i - i) * velocity1)
                for i in range(last_i, len(tracking_centers[track_id])):
                        tracking_centers[track_id][i] = tracking_centers[track_id][last_i] + ((i - last_i) * velocity2)
                        tracking_centers[track_id][i] = tracking_centers[track_id][i].tolist() # - ((first_i - i) * velocity1)
                        """
                extrapolate_center = tracking_centers[track_id][first_i] - (first_i * velocity)
                                    
                tracking_meta[track_id] = {
                    # "speed": speed,
                    "extrap_center": extrapolate_center
                }
            #if scene_token == "798e8504b4364d378270333a349ef508":
            #    breakpoint()
                
            dataset_name = "nuscenes"        
        
            os.makedirs(f"logs/{dataset_name}/{exp}/track_output__rerun/{scene_token}/{X['name']}", exist_ok=True)
            print("Save into ", f"logs/{dataset_name}/{exp}/track_output__rerun/{scene_token}/{X['name']}")
            with open(f"logs/{dataset_name}/{exp}/track_output__rerun/{scene_token}/{X['name']}/coordinates.json", "w") as f:
                import json
                json.dump([v for k,v in sorted(tracking_centers.items(), key=lambda item: item[0]) if k in tracking_meta],f)
                            
            with open(f"logs/{dataset_name}/{exp}/track_output__rerun/{scene_token}/{X['name']}/track.json", "w") as f:
                
                import json
                codes = []
                for ii, track_id in enumerate( sorted(list(tracking_centers.keys())) ): # ummm wy
                    if track_id not in tracking_meta:
                        continue
                    p = f"""new Car offset by ({tracking_meta[track_id]["extrap_center"][0]}, {tracking_meta[track_id]["extrap_center"][1]}),
        with blueprint "vehicle.lincoln.mkz_2020",
        with color Color(0.3,0.3,0.3)
                    """ 
                    codes.append({"code": p})
                json.dump(codes,f)
nuscenes()

import os

import tqdm

list_dir = sorted(os.listdir('/work/users/j/l/jleung18/waymo_1/validation'))