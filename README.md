# Road2Code

This is the code for Road2Code, accepted to International Conference on Neuro-symbolic Systems, 2025.

## A Note on `git clone`

Use `git clone --recurse-submodules https://github.com/road-2-code/road-2-code` to get the code including the files from the `Scenic` submodule.

## Training the Model

Place your training dataset under `data/training_` 

`cd scenic_finetuning.py`
`python model.py`

Choose the model to run inference in `inference.py`, then run:
`python inference.py`

## Rendering Scenes

Ensure your generated programs is in a folder, for example, `output/x.scenic` where `x` is a number, then
Render scenes, using this example:

`./carla/CarlaUE4.sh -RenderOffScreen -carla-rpc-port=5000 -nosound -carla-server -carla-server-timeout=10000ms -opengl`

`python run_renderer.py --start 0 --end 22 --port 5000 --render-output --folder output_llm --multiview --config waymo`

where `start` and `end` are the number ranges of the selected scenes to render.