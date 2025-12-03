# Policy Contrastive Decoding for Robotic Foundation Models (LeRobot Evaluation)

Official implementation of the paper "[Policy Contrastive Decoding for Robotic Foundation Models](https://arxiv.org/abs/2505.13255)".

> Note: This is the LeRobot evaluation of our work. For SIMPLER experiments, please check out [PCD](https://github.com/Koorye/PCD).

> Note: We are doing our best to improve this work. If you have any questions or suggestions, please feel free to create an issue in this repo or contact us at shihan.wu.koorye@outlook.com.

[[Project]](https://koorye.github.io/proj/PCD) [[ArXiv]](https://arxiv.org/abs/2505.13255) [[PDF]](https://arxiv.org/pdf/2505.13255) [[PCD]](https://github.com/Koorye/PCD)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Koorye/PCD-LeRobot.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -e .
   # install pytorch
   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

   # install grounded sam 2
   cd src/lerobot/policies/contrast_utils/grounded_sam_2
   pip install -e .
   pip install --no-build-isolation -e grounding_dino

   # install inpaint-anything
   pip install -r src/openpi/models/contrast_utils/inpaint_anything/lama/requirements.txt

   # install other dependencies
   pip install transformers==4.40.1
   pip install numpy==1.26.4
   pip install hydra-core==1.3.2
   ```

3. Download the pre-trained checkpoints:
   ```bash
   mkdir pretrained
   cd pretrained/

   # download sam2
   wget "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
   
   # inpaint_anything
   # PLEASE download 'big-lama' from the google drive: https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg?usp=sharing

   # install huggingface-cli
   pip install huggingface_hub
   
   # install grounding-dino
   current_dir=$(pwd)
   cd ~/.cache/huggingface/hub
   huggingface_cache_dir=$(pwd)
   cd $current_dir

   huggingface-cli download "IDEA-Research/grounding-dino-base"
   ln -s ${huggingface_cache_dir}"/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb" ${current_dir}"/grounding-dino-base"
   ```

## Usage

1. Collecting Data. 
   We provide an example script to collect data for the task "place the bowl on the plate" using a Agilex Piper robot & ROS camera with a follower-leader teleoperation setup.
   ```bash
   lerobot-record \
       --robot.type=piper_follower \
       --robot.can=can0 \
       --robot.init_type=none \
       --robot.cameras="{ front: {type: ros_camera, node_name: front, topic: "/ob_camera_01/color/image_raw", fps: 30, width: 640, height: 480} }" \
       --robot.visualize=False \
       --robot.id=follower \
       --dataset.repo_id=piper/place_the_bowl_on_the_plate \
       --dataset.num_episodes=50 \
       --dataset.single_task="place the bowl on the plate" \
       --dataset.push_to_hub=False \
       --teleop.type=piper_leader \
       --teleop.can=can0 \
       --teleop.id=leader \
       --resume=True
   ```

2. Training Diffusion Policy.
   We provide an example script to train a diffusion policy on the collected dataset.
   ```bash
   lerobot-train \
       --dataset.repo_id=piper/place_the_bowl_on_the_plate \
       --dataset.video_backend=pyav \
       --policy.type=diffusion \
       --policy.n_obs_steps=2 \
       --policy.horizon=32 \
       --policy.n_action_steps=16 \
       --policy.push_to_hub=False \
       --output_dir=runs/place_the_bowl_on_the_plate/diffusion \
       --job_name=place_the_bowl_on_the_plate_diffusion \
       --resume=False \
       --seed 0 \
       --batch_size=16 \
       --num_workers=8 \
       --steps=10000 \
       --log_freq=100 \
       --eval_freq=1000 \
       --save_freq=1000 
   ```

3. Inference Baseline.
   We provide an example script to run inference on the trained model with the Agilex Piper robot.
   ```bash
   python src/lerobot/scripts/inference_local.py \
       --robot.type=piper \
       --robot.can=can0 \
       --robot.cameras="{ front: {type: ros_camera, node_name: front, topic: "/ob_camera_01/color/image_raw", fps: 30, width: 640, height: 480} }" \
       --robot.id=piper \
       --robot.visualize=False \
       --use_pcd=False \
       --pretrained_path="runs/place_the_bowl_on_the_plate/diffusion/checkpoints/last/pretrained_model" \
       --repo_id="piper/place_the_bowl_on_the_plate" \
       --result_dir="results/" \
       --frequency 10
   ```
   You can press `q` to stop the inference and press `y/n` to decide whether the trial is successful.

4. Inference w/ PCD.
   We provide an example script to run inference on the trained model with the Agilex Piper robot using PCD.
   ```bash
   python src/lerobot/scripts/inference_local.py \
       --robot.type=piper \
       --robot.can=can0 \
       --robot.cameras="{ front: {type: ros_camera, node_name: front, topic: "/ob_camera_01/color/image_raw", fps: 30, width: 640, height: 480} }" \
       --robot.id=piper \
       --robot.visualize=False \
       --use_pcd=True \
       --pretrained_path="runs/place_the_bowl_on_the_plate/diffusion/checkpoints/last/pretrained_model" \
       --repo_id="piper/place_the_bowl_on_the_plate" \
       --result_dir="results/" \
       --frequency 10
   ```
   PCD use the same pretrained model without any finetuning. You can press `q` to stop the inference and press `y/n` to decide whether the trial is successful. 

## Experiments

### Training Settings

We designed a "place bowl on plate" task using an AGILEX PIPER robotic arm, collected 50 demonstration trajectories and trained a Diffusion Policy from scratch. The model utilized a ResNet18 visual backbone and was implemented using the the LeRobot framework.

| Hyperparameter       | Value                      |
| ---------------------|----------------------------|
| Steps                | 10000                      |
| Batch Size           | 16                         |
| Learning Rate        | 1e-4                       |
| Optimizer            | Adam                       |
| Observation Window   | 2                          |
| Action Chunk Size    | 16                         |

### PCD Inference Settings

> See `src\lerobot\policies\diffusion\modeling_diffusion.py` for more details.

| Hyperparameter       | Value                      |
| ---------------------|----------------------------|
| $\alpha$             | 1.0                        |
| $N$                  | 24                         |

### Results

| Model | Success Rate (%) |
|-------|------------------|
| Diffusion Policy | 15    |
| **+PCD (Ours)** | **45** |

| Diffusion Policy | +PCD (Ours) |
| -----------------|-------------|
| ![Diffusion](examples/place_the_bowl_on_the_plate_failed_20251125_164540.gif) | ![PCD](examples/place_the_bowl_on_the_plate_success_20251125_200112.gif) |

## Acknowledgements

This code is built upon the [LeRobot](https://github.com/huggingface/lerobot) framework. 
We would like to thank the LeRobot team for their open-source contributions.