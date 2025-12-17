<h2 align="center">
  <!-- <img src="docs/gentle_humanoid_logo.png" height="24" style="vertical-align: bottom; margin-right: 0px;" /> -->
  <a href="https://gentle-humanoid.axell.top">GentleHumanoid: Learning Upper-body Compliance for Contact-rich Human and Object Interaction</a>
</h2>

[![Home Page](https://img.shields.io/badge/Project-Website-C27185.svg)](https://gentle-humanoid.axell.top/#/) 
[![arXiv](https://img.shields.io/badge/Arxiv-2511.04679-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2511.04679) 
[![Video](https://img.shields.io/badge/Video-Demo-FF0000.svg?logo=youtube)](https://www.youtube.com/watch?v=rF6N2o0IQJg)
[![Online Demo](https://img.shields.io/badge/Online-Demo-3B82F6.svg?logo=demo)](https://gentle-humanoid.axell.top/#/demo)

This is an official implementation of GentleHumanoid, more details please check our [Project](https://gentle-humanoid.axell.top) page. 

This repo provides the codebase for training and evaluating GentleHumanoid policies. For deployment on real robots, please check [here](https://github.com/Axellwppr/gentle-humanoid/).

## Installation

1. Create a Conda environment.
```
conda create -n gentle python=3.10
conda activate gentle
```

2. Install Torch.
```
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

3. Install Isaac Sim. Support Ubuntu 22.04 only. For 20.04, please check [here](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/binaries_installation.html).
```
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
# Test Isaacsim
isaacsim
```
4. Install Isaac Lab.
```
cd <where you want to install IsaacLab>
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.2.0
./isaaclab.sh -i none
```

5. Install GentleHumanoid.
```
cd <where you want to install repo>
git clone https://github.com/Axellwppr/gentle-humanoid-training
cd gentle-humanoid-training
pip install -e .
```

## Motion Dataset Preparation

### Retargeting with GMR

We use GMR to retarget the [AMASS](https://amass.is.tue.mpg.de/), [Inter-X](https://github.com/liangxuy/Inter-X), and [LAFAN](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) datasets. The output format is a dataset containing a series of npz files with the following fields:

- `fps`: Frame rate
- `root_pos`: Root position
- `root_rot`: Root rotation in quaternion format (xyzw)
- `dof_pos`: Degrees of freedom positions
- `local_body_pos`: Local body positions
- `local_body_rot`: Local body rotations
- `body_names`: List of body names
- `joint_names`: List of joint names

You can use the [modified version of GMR](https://github.com/Axellwppr/GMR) to directly export npz files that meet the requirements.

Note: When processing the Inter-X dataset, you need to handle the coordinate axis definition differences from AMASS and the missing framerate. Please refer to [here](https://github.com/Axellwppr/GMR/blob/c8970d755519d9bb6e79786a3fb43649b71198fa/scripts/smplx_to_robot_dataset.py#L155) for details.

You should organize the processed datasets in the following structure:
```
<dataset_root>/
    AMASS/ACCAD/Female1General_c3d/A1_-_Stand_stageii.npz
    ...
    LAFAN/walk1_subject1.npz
    ...
    InterX/G001T000A000R000/P1.npz
```

### Dataset Building

Modify `DATASET_ROOT` in `generate_dataset.sh` to point to your dataset root directory, then run the script to generate the dataset:
```
bash generate_dataset.sh
```

The dataset will be generated in the `dataset/` directory, and the code will automatically load these datasets. You can also use the `MEMATH` environment variable to specify the dataset root path.

## Training

You can use the provided `train.sh` script to run the full training pipeline. Modify the global configuration section in `train.sh` to set your WandB account and other parameters, then run:

```bash
bash train.sh
```

Under standard settings, training takes approximately 5 hours on 4× A100 GPUs.
If GPU memory is constrained, it is recommended to appropriately tune the `NPROC` and `num_envs` parameters in `train.sh` and `cfg/task/G1/G1.yaml`, respectively.
Such adjustments may increase training time and could affect training performance to some extent.

## Evaluation

```bash
python scripts/eval.py --run_path ${wandb_run_path} -p # p for play
python scripts/eval.py --run_path ${wandb_run_path} -p --export # export the policy to onnx (sim2real)
```