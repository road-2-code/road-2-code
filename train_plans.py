# from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import time

import torch
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torchvision

import sys

from utils.program_synthesis import write_program
from utils.functions import get_iou
from utils.config import HEIGHT, WIDTH
from utils.dataset import ImageDataset, KITTIDataset, NuscenesDataset
from utils.segmentation import Runner
from utils.training import Trajectories, TrajectoryPoint

import wandb

from utils.model import EgoLocationModel, Model, UniADLocationModel

import tqdm
import argparse

import yaml

from utils.config import IMAGE_DIR, TRAINING_DATA_DIR, SEG_DIR


import numpy as np

def find_intervals(sorted_list, input_points):
    # Convert to numpy arrays
    sorted_arr = np.array(sorted_list)
    points_arr = np.array(input_points)
    
    # Use searchsorted to find indices where input points would fit
    indices = np.searchsorted(sorted_arr, points_arr, side='left')
    
    # Initialize result list to store intervals
    results = []
    
    for idx, point in zip(indices, points_arr):
        if idx == 0:
            results.append(f"({float('-inf')}, {sorted_arr[0]}]")
        elif idx == len(sorted_arr):
            results.append(f"({sorted_arr[-1]}, {float('inf')})")
        else:
            results.append(f"({sorted_arr[idx - 1]}, {sorted_arr[idx]}]")
    
    return results



parser = argparse.ArgumentParser()
parser.add_argument('--depth-model', type=str, default="")
parser.add_argument('--loss', type=str, default="l1")
parser.add_argument('--exp', type=str, default="exp", help="experiment name")
parser.add_argument('--debug', action="store_true", help="enable debug")
parser.add_argument('--disable-wandb', action="store_true", help="enable debug")
parser.add_argument('--dataset', type=str, help="config file path")
parser.add_argument('--infer', action="store_true", help="")
parser.add_argument('--model', type=str, default="model_0")
parser.add_argument('--epoch-count', type=int, default=40)
parser.add_argument('--model-i', type=int, default=-1)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--use-direction', action="store_true")
parser.add_argument('--use-direction-2', action="store_true")
parser.add_argument('--margin', type=float, default=5)
parser.add_argument('--out-path', type=str, default="no")

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}


# encoder = 'vits' # or 'vitb', 'vits'
# depth_anything = DepthAnything(model_configs[encoder])
# depth_anything.load_state_dict(torch.load(f'./checkpoints/depth_anything_{encoder}14.pth'))
# depth_anything.to(device)

# for param in depth_anything.parameters():
#     param.requires_grad = False

def save_np_image(array, file_name, title="", rects=[], text=[], rects_2=[], use_annot=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1,1,figsize=(50,25))
    
    im = ax.imshow(array)
    fig.colorbar(im)
    ax.set_title(title)

    if use_annot:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                ax.text(j,i,f"{array[i,j]:.2f}")
    
    for rect in rects:
        # print(rects)
        (x_min,x_max),(y_min,y_max)=rect
        # x_min/=3
        # y_min/=3
        # x_max/=3
        # y_max/=3
        rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    for rect in rects_2:
        # print(rects)
        (x_min,x_max),(y_min,y_max)=rect
        # x_min/=3
        # y_min/=3
        # x_max/=3
        # y_max/=3

        rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    for t in text:
        x,y,string = t
        ax.text(x,y,string,color='white',fontsize=10)

    fig.savefig(file_name)
    
    fig.clear() # avoid memleak
    
    # Clear the current axes.
    plt.cla() 
    # Clear the current figure.
    plt.clf() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)
    import gc
    gc.collect()


class ScenicTrainer:
    def __init__(self,exp_name, criterion, config, dataset_config, lr, uniad_model, args):
        self.exp_name = exp_name
        self.model = Model(config['model'], args)
        self.model.to(device)
        self.ego_location_model = EgoLocationModel()
        self.ego_location_model.to(device)
        self.criterion = criterion
        self.args = args

        self.dataset_config = dataset_config        
        self.log_dir = "kitti/logs"
        
        self.image_width = WIDTH
        self.image_height = HEIGHT
        
        self.test_step = 0
        self.train_step = 0
        
        self.best_loss = 0

        self.uniad_model = uniad_model
        self.uniad_model.requires_grad = False
        self.uniad_location_model = UniADLocationModel()
        self.uniad_location_model.to(device)

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) 
                                          + list(self.ego_location_model.parameters()) 
                                          + list(self.uniad_location_model.parameters()), lr=lr)
        self.lane_criterion = torch.nn.CrossEntropyLoss()
        
    def infer_depth_in_inference(self, data_batch):
        import time
        a = time.time()

        
        image_batch_ = torch.stack([data_batch[i]['image'] for i in range(len(data_batch))])
        image_batch_ = torch.permute(image_batch_, (0,3,1,2)).float().to(device)
        
        lane_logits_batch = []
        ego_lane_logits_batch = []
        direction_logits_batch = []
        estim_depths_batch = []
        # depth = length of depth batch.
        for i, (data, image) in enumerate(zip(data_batch,image_batch_)):
            #ego_lane_logits = self.ego_location_model(road_seg_mask_section, road_depth_section)[0]
            ego_perception_logits = self.ego_location_model(image[None,...])

            
        return None, estim_depths_batch, ego_lane_logits_batch, lane_logits_batch, direction_logits_batch

    def save_model(self,i):
        os.makedirs(f"logs/{self.exp_name}", exist_ok=True)
        torch.save({
                'model_state_dict': self.model.state_dict(),
                'ego_state_dict': self.ego_location_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, f"logs/{self.exp_name}/model.pth")
        torch.save({
                'model_state_dict': self.model.state_dict(),
                'ego_state_dict': self.ego_location_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, f"logs/{self.exp_name}/model_{i}.pth")


    def load_model(self, i):
        return
        if i != -1:
            params = torch.load(f"logs/{self.exp_name}/model_{i}.pth")
        else:
            params = torch.load(f"logs/{self.exp_name}/model.pth")
        self.model.load_state_dict( params['model_state_dict'] )
        self.model.args = args
        self.ego_location_model.load_state_dict( params['ego_state_dict'] )
        self.optimizer.load_state_dict( params['optimizer_state_dict'] )
        
        # self.optimizer = optimizer
    def infer_depth(self, data_batch, mode, train_loss):
        """
        step: step count for logging
        batch_id: batch id for logging
        databatch: a sequence (eg a short sequence) of frames to process
        """
    
        bce_loss = torch.nn.BCEWithLogitsLoss()
        
        image_batch_ = torch.stack([data_batch[i]['image'] for i in range(len(data_batch))])
        #image_batch_ = torch.permute(image_batch_, (0,3,1,2)).float().to(device)
        h, w = image_batch_.shape[2], image_batch_.shape[3]

        gnd_lane_ids_batch = [data_batch[i]["lane_idx"] for i in range(len(data_batch))]
        direction_ids_batch = [data_batch[i]["direction"] for i in range(len(data_batch))]
        # For each image in the batch, we have a list of perceptions. This list is a list of booleans.
        # Each boolean denotes whether a perception is "true" or "false"
        # The format is
        # [is_vehicle_in_fov(20 meters), is_vehicle_in_fov(40 meters)... is_vehicle_in_fov(100 meters),
        # is_vehicle_in_lane(20 meters), is_vehicle_in_lane(40 meters)... is_vehicle_in_lane(100 meters),
        # ]
        perceptions_batch = []# [data_batch[i]["perceptions"] for i in range(len(data_batch))]
        
        # predicted logits (for code o fill in)
        lane_logits_batch = []
        ego_lane_logits_batch = []
        direction_logits_batch = []
        estim_depths_batch = []
        


        return None, estim_depths_batch, ego_lane_logits_batch, lane_logits_batch, direction_logits_batch, torch.tensor([0])

    program_criterion = torch.nn.CrossEntropyLoss()

    def run_iteration(self, step, data_batch, mode="train", train_loss=True):
        """
        step: counts the current step for logging
        batch id: counts the current step for logging
        data_batch: Contains info about a batch multiple videos. Each video has a program and a sequence of frames
        """
        
        total_loss_data = 0
        counter = 0
        
        import copy
        
        
        # For each video that we find
        for i, X in enumerate(data_batch):
            uniad_model = copy.deepcopy(self.uniad_model)
            sequence_data = X['frames']
            from UniAD.run_uniad import custom_test

            print(X['frames'][0]['image'].shape)
            input_data = [X['frames'][i]['image'].permute((0,1,4,2,3)).float().to(device)
                                                          for i in range(len(X['frames']))]
            
            tracking_centers = {}
            
            length = len(input_data)

            for i, image in enumerate(input_data):
                # timestamp is important for tracking
                result = custom_test(uniad_model, {'img':[image]}, i * 0.5)
                print(image.mean(), "<<< mean")
                
                current_frame =  X['frames'][0]

                #print(result)
        
                tracking = result[0]
                #ego_location, ego_rotation = current_frame['transforms'][0]
            
                bev_tracking_data = tracking['pts_bbox']
                track_ids = tracking['track_ids']
                labels = tracking['labels_3d']
                
                is_cars = labels == 0

                boxes_3d = tracking['boxes_3d'][is_cars]
                track_ids = track_ids[is_cars]

                for track_id in track_ids:
                    track_id = track_id.item()
                    if track_id not in tracking_centers:
                        tracking_centers[track_id] = [None] * length

                # bounding_boxes = [get_bbox_vertices([b[1], b[0]], [b[4]/2, b[3]/2], 0) for b in boxes_3d]
                
                vehicle_centers = [np.array([b[0].item(), b[1].item()] ) for b in boxes_3d]
                
                
                for center, track_id in zip(vehicle_centers, track_ids):
                    track_id = track_id.item()
                    tracking_centers[track_id][i] = center
                
                
                # Get the coordinates of lanes labeled as 1 (has a lane)
                y_coords, x_coords = np.where((bev_tracking_data['lane_score'][:,50:150,50:150].cpu().numpy() > 0.7).any(0))
                x_coords = x_coords[:,None]
                
                x_coords -= 50 # centered about zero.

                from sklearn.mixture import GaussianMixture
                import matplotlib.pyplot as plt

                # ================================
                # Fit the Gaussian Mixture Model
                gmm = GaussianMixture(n_components=3, covariance_type='full')
                gmm.fit(x_coords)

                # Predict cluster labels
                labels_gmm = gmm.predict(x_coords)
                output_intervals = find_intervals(gmm.means_.squeeze(), [c[0] for c in vehicle_centers])
                
                loc = torch.tensor([[0., 0.]])
                loc = loc.to(device)
                
                # centers = [v for v in boxes_3d.tensor[:,0:2].numpy() if abs(v[0]) < 3]
                # if len(centers) > 0:
                #     centers = np.stack(centers)
                # else:
                #     centers = np.array([[]])
                # print(centers)
                centers = boxes_3d.tensor[:,0:2]

                
                import matplotlib.pyplot as plt
                os.makedirs(f"logs/nuscenes/{args.exp}/dbg/{X['name']}", exist_ok=True)
                plt.imshow(bev_tracking_data['drivable'].cpu())
                plt.savefig(f"logs/nuscenes/{args.exp}/dbg/{X['name']}/" + f"{i}_road.png")

                
                # ===== debug
                if True:
                    os.makedirs(f"logs/carla/{args.exp}/plts/", exist_ok=True)
                    plt.cla()
                    plt.imshow(bev_tracking_data['drivable'].cpu())
                    # plt.imshow((bev_tracking_data['drivable']).cpu())
                    plt.vlines(x=gmm.means_+100, ymin=0, ymax=200)
                    
                    plt.scatter(2*centers[:,0]+100, 2*centers[:,1]+100, c=track_ids, cmap='tab10')

                    for j, label in enumerate(track_ids):
                        plt.text(2*centers[j,0]+100, 2*centers[j,1]+100, str(label.item()), fontsize=9)
                    
                    plt.savefig(f"logs/carla/{args.exp}/plts/{X['name']}__{i}.png")
            

            speeds = {}
            all_track_ids = tracking_centers.keys()
            
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
                        if abs(track[0]) > args.margin * 0.75:
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
                
                print(velocity)
                    
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
            
        
            os.makedirs(f"logs/nuscenes/{args.exp}/track_output__10m_margin/{X['name']}", exist_ok=True)
            with open(f"logs/nuscenes/{args.exp}/track_output__10m_margin/{X['name']}/coordinates.json", "w") as f:
                import json
                json.dump([v for k,v in sorted(tracking_centers.items(), key=lambda item: item[0]) if k in tracking_meta],f)
                        
            with open(f"logs/nuscenes/{args.exp}/track_output__10m_margin/{X['name']}/track.json", "w") as f:

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

            del uniad_model
            # with open(f"logs/nuscenes/{args.exp}/track_output/{X['name']}/meta.json", "w") as f:
            #     import json
            #     l = []
            #     for ii,  track_id in enumerate ( tracking_centers.keys() ):
            #         if track_id not in tracking_meta:
            #             continue
            #         l.append({"center": str(list(tracking_meta[track_id]["extrap_center"]))})
            #     json.dump(l,f)
                        
            # with open(f"logs/nuscenes/{args.exp}/track_output/{X['name']}/track.json", "w") as f:

            #     f.write("[")
            #     for ii, track_id in enumerate( tracking_centers.keys() ):
            #         if track_id not in tracking_meta:
            #             continue
            #         p = f"[INST] Follow lane at target speed {tracking_meta[track_id]['speed']}. [/INST]" 
            #         # starting from {tracking_meta[track_id]["extrap_center"]}"
            #         f.write("{\"desc\":\" " + p + "\"}")
            #         if ii < len(tracking_centers) - 1:
            #             f.write(",")
            #     f.write("]")
        
        
        return 0 #total_loss_data / float(counter)
    


def main():
    exp = args.exp

    LOG_DIR = 'logs/carla'
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR + f"/{exp}/train/segs", exist_ok=True)
    os.makedirs(LOG_DIR + f"/{exp}/train/depths", exist_ok=True)
    os.makedirs(LOG_DIR + f"/{exp}/train/programs", exist_ok=True)

    os.makedirs(LOG_DIR + f"/{exp}/test/segs", exist_ok=True)
    os.makedirs(LOG_DIR + f"/{exp}/test/depths", exist_ok=True)
    os.makedirs(LOG_DIR + f"/{exp}/test/programs", exist_ok=True)

    infer_dir = "logs/kitti"

    os.makedirs(infer_dir + f"/{exp}/test/segs", exist_ok=True)
    os.makedirs(infer_dir + f"/{exp}/test/depths", exist_ok=True)
    os.makedirs(infer_dir + f"/{exp}/test/programs", exist_ok=True)


    with open("config/config.yaml") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


    with open("config/kitti.yaml") as stream:
        try:
            dataset_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        
    import torch
    if args.loss == 'l1':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'l2':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError("Invalid --loss")
    
    
    torch.manual_seed(780)
    
    from UniAD.run_uniad import main as uniad_main
    uniad_tracker = uniad_main()
    uniad_tracker.to(device)
    


    wandb.login()

    if args.disable_wandb or args.infer:
        run = wandb.init(
            # Set the project where this run will be logged
            project="Carla",
            name=f"neurosymbolic_programming_{args.exp}",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "epochs": args.epoch_count,
            },
            mode="disabled"
        )
    else:
        run = wandb.init(
            # Set the project where this run will be logged
            project="Carla",
            name=f"neurosymbolic_programming_{args.exp}",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "epochs": args.epoch_count,
            },
        )
    trainer = ScenicTrainer(exp, criterion, config, dataset_config, args.lr, uniad_tracker, args)

    N = 40# epochs
    
    object_detection_model = None

    kitti_dataset = KITTIDataset()
    #dataset = ImageDataset('output', dataset_config, args.dataset)
    dataset = NuscenesDataset(multiview=True)
    dataset = torch.utils.data.Subset(dataset, range(0,10000))
    
    from torch.utils.data import random_split
    generator = torch.Generator().manual_seed(42)
    import math
    data_len = len(dataset)
    train_dataset, test_dataset, val_dataset = dataset, dataset, dataset #random_split(dataset, [math.ceil(0.7*data_len),
                                                                      #math.ceil(0*data_len),
                                                                      #data_len-math.ceil(0.7*data_len)-math.ceil(0*data_len)], generator=generator)
    # test_dataset = dataset
    
    # no shuffle vvvvvv
    loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    kitti_loader = DataLoader(kitti_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    
    start_i = 0

    epoch_count = args.epoch_count
    
    # load existing model and run inference
    if args.infer:
        trainer.load_model(args.model_i)
        # for batch_id, data_batch in tqdm.tqdm(enumerate(loader), total=len(loader)):
        #     with torch.no_grad():
        #         trainer.run_iteration(batch_id, batch_id, data_batch, runner, program_synthesizer, object_detection_model, program_optimizer, mode="train", train_loss=False)


        #for batch_id, data_batch in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        #    with torch.no_grad():
        #        trainer.run_iteration(batch_id, batch_id, data_batch, runner, program_synthesizer, object_detection_model, program_optimizer, mode="test")

        for batch_id, data_batch in tqdm.tqdm(enumerate(kitti_loader), total=len(kitti_loader)):
            with torch.no_grad():
                trainer.run_iteration(batch_id, data_batch, mode="infer")
            # test_output_file.write(f"Total Test Loss {epoch}: {test_loss}")
        
    else:
        test_output_file = open(LOG_DIR + f"/{exp}/loss.txt", "a")
        if args.model_i != -1:
            print("Loading", args.model_i)
            trainer.load_model(args.model_i)
            start_i = int(args.model_i)
         
            
        for epoch in range(start_i, epoch_count):
            print("epoch i:", epoch)
            train_loss = 0
            start_time = time.time()
            print("Training epoch", epoch)
            for batch_id, data_batch in tqdm.tqdm(enumerate(loader), total=len(loader)):

                train_loss += trainer.run_iteration(epoch * len(loader) + batch_id, data_batch)

                if batch_id > 10:
                    raise ""
            end_time = time.time()
            print("Time iterated", start_time - end_time)

            train_loss /= len(loader)
            print(f"Total Training Loss {epoch} is {train_loss}")
            test_output_file.write(f"Total Training Loss {epoch}: {train_loss}\n")
            test_output_file.flush()
            
            val_loss = 0
            print("Val epoch", epoch)
            for batch_id, data_batch in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):
                with torch.no_grad():
                    val_loss += trainer.run_iteration(epoch * len(test_loader) + batch_id, data_batch,  mode="val")
            val_loss /= len(val_loader)
            print(f"Total Val Loss {epoch} is {val_loss}")
            test_output_file.write(f"Total Val Loss {epoch}: {val_loss}\n")
            test_output_file.flush()
            test_loss = 0
            print("Testing epoch", epoch)
            for batch_id, data_batch in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
                with torch.no_grad():
                    test_loss += trainer.run_iteration(epoch * len(test_loader) + batch_id, data_batch,  mode="test")
            test_loss /= len(test_loader)
            print(f"Total Test Loss {epoch} is {test_loss}")
            test_output_file.write(f"Total Test Loss {epoch}: {test_loss}\n")
            test_output_file.flush()
            
            if epoch % 1 == 0:
                trainer.save_model(epoch)

            #for batch_id, data_batch in tqdm.tqdm(enumerate(kitti_loader), total=len(kitti_loader)):
            #    print("Infer epoch", epoch)
            #    with torch.no_grad():
            #        trainer.run_iteration(epoch * len(kitti_loader) + batch_id, batch_id, data_batch, runner, program_synthesizer, object_detection_model, program_optimizer, mode="infer")
            #        # print("elapsed", b-a)
main()
