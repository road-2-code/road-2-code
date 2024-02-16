IMAGE_DIR = '/work/users/j/l/jleung18/carla/data/images'
SEG_DIR = '/work/users/j/l/jleung18/carla/data/output_seg'
DEPTH_DIR = '/work/users/j/l/jleung18/carla/data/depth'

TRAINING_DATA_DIR = '/work/users/j/l/jleung18/carla/data/training_data'


WIDTH = 1242
HEIGHT = 375

road_configurations = [{
   'map': 'Town04',
   'isect': 10,
   'lanes': [(False,0),(False,1),(False,2),(False,3), (True,5),(True,6),(True,7),(True,8)]
}, {
    'map': 'Town03',
    'isect': 0,
    'lanes': [(False,0),(False,1),None,None,None,None,(True,0),(True,1)]
}]
