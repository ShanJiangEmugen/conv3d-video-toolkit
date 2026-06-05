[![Release](https://img.shields.io/github/v/release/ShanJiangEmugen/conv3d-video-toolkit?style=flat-square)](https://github.com/ShanJiangEmugen/conv3d-video-toolkit/releases/latest)
[![License: MIT](https://img.shields.io/github/license/ShanJiangEmugen/conv3d-video-toolkit?style=flat-square)](LICENSE)
[![Issues](https://img.shields.io/github/issues/ShanJiangEmugen/conv3d-video-toolkit?style=flat-square)](https://github.com/ShanJiangEmugen/conv3d-video-toolkit/issues)
[![Stars](https://img.shields.io/github/stars/ShanJiangEmugen/conv3d-video-toolkit?style=flat-square)](https://github.com/ShanJiangEmugen/conv3d-video-toolkit/stargazers)


This repository provides a complete and modular pipeline for video-based behavior / action classification using 3D Convolutional Neural Networks (Conv3D).
It includes:

- Data preprocessing utilities
- Video dataset splitting tools
- Training and fine-tuning scripts
- Keras 3D ConvNet models
- Sliding-window video inference
- CSV-based probability outputs
- Optional PyTorch experimental modules
- Jupyter notebook demos for analysis and visualization

The project is structured so that each component can also be used independently.

# Features

- Flexible video generators for training & inference
- C3D-style ConvNet implemented in Keras / TensorFlow
- Full training pipeline (from scratch or transfer learning)
- Fine-tuning with custom output layers
- Sliding-window inference over entire videos
- Automatic top-model symlink (`lastest_model.h5`)
- Video preprocessing (resize, downsample)
- Experimental PyTorch implementation
- Jupyter notebooks for inference visualization


# Installation
1. Clone the repository
```bash
git clone https://github.com/ShanJiangEmugen/conv3d-video-toolkit.git
cd <repo-name>
```

2. Install dependencies
```bash
pip install -r requirements.txt 
```

# Usage Overview
### 1. Prepare the dataset     
  Organize videos into class-named directories:
  ```
  dataset_root/
      ├── class_A/
      │     ├── video1.avi
      │     ├── video2.avi
      ├── class_B/
            ├── video3.avi
            ├── video4.avi
  ```
  
  Split into train/test/val:
  ```bash
  python data_prep/split_dataset.py
  ```
  
  (Optional) Resize video resolution:
  ```bash
  python data_prep/resize_clips.py
  ```

### 2. Train a Conv3D model
  From scratch:
  ```bash
  python train/train_ss.py
  ```
  
  Fine-tune an existing model:
  ```bash
  python train/fine_tune.py
  ```
  
  Continue training on the latest saved model:
  ```bash
  python train/re_train_ss.py
  ```
  
  After each training session, a symlink:
  ```bash
  models/lastest_model.h5
  ```
  
  always points to the most recent .h5 model. 

### 3. Run inference on new videos
```bash
python inference/infer.py --infer_dir path/to/videos
```

This produces:
```
predictions/<video_name>_prediction.csv
```
with class probabilities for each sliding-window segment.

### 4. Explore results in notebooks

Open:

- `notebooks/infer_demo.ipynb`
- `notebooks/infer_analysis_demo.ipynb`

or the experiment notebooks under:
```
experiments/conv3d_inception/
```

# Experimental PyTorch Version
The `experimental/` folder contains early prototypes of a PyTorch implementation:

- Basic 3D CNN (`torch_network.py`)
- Video segment dataset (`torch_video_gen.py`)

These are included for reference and are not **production-ready**.

# Dependencies
Major frameworks used:

- TensorFlow / Keras
- OpenCV
- ffmpeg-python
- NumPy, Pandas
- tqdm
- PyTorch (optional for experiments)

Full list in `requirements.txt`.
