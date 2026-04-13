# Assignment 02 - DIP with PyTorch

This repository/report contains the implementation and experiment record for `Assignment 02 - DIP with PyTorch`.

## Requirements

### Environment

- OS: Windows
- Python: 3.10
- Conda environment: `gdl_env`
- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`
- CUDA: `13.0`

### Install requirements

```bash
conda activate gdl_env
pip install torch gradio pillow numpy opencv-python
```

### Dataset preparation

Download the Facades dataset and generate file lists:

```bash
conda activate gdl_env
cd Pix2Pix
python download_facades_dataset.py
```

This script will:

- download `facades`
- extract it to `datasets/facades/`
- generate `train_list.txt`
- generate `val_list.txt`

## Training

### 1. Poisson Image Editing

Poisson blending is solved by optimization instead of standard offline training.

Run:

```bash
conda activate gdl_env
python run_blending_gradio.py
```

Workflow:

1. Upload the foreground image.
2. Click several points to define a polygon.
3. Click `Close Polygon`.
4. Upload the background image.
5. Adjust `Horizontal Offset` and `Vertical Offset`.
6. Click `Blend Images` to optimize the blended result.

Implemented functions:

- `create_mask_from_points`
- `cal_laplacian_loss`

### 2. Pix2Pix

Train the FCN-based Pix2Pix model:

```bash
conda activate gdl_env
cd Pix2Pix
python train.py
```

Quick smoke test:

```bash
python train.py --train_list train_list.txt --val_list val_list.txt --num_epochs 1 --batch_size 4 --num_workers 0 --max_train_steps 2 --max_val_steps 1 --save_every 1 --output_dir checkpoints
```

Network design:

- Encoder: `Conv2d + BatchNorm2d + LeakyReLU`
- Decoder: `ConvTranspose2d + BatchNorm2d + ReLU`
- Output activation: `Tanh`

Training setup:

- Optimizer: Adam
- Learning rate: `0.001`
- Loss: `L1Loss`
- Scheduler: `StepLR(step_size=200, gamma=0.2)`

## Evaluation

### 1. Poisson Image Editing

Evaluation is qualitative and based on visual inspection:

- whether the pasted region matches the background naturally
- whether the boundary is smoother than naive copy-paste
- whether different polygon regions and offsets still produce stable blending

### 2. Pix2Pix

An evaluation script is included:

```bash
conda activate gdl_env
cd Pix2Pix
python evaluate.py --checkpoint checkpoints/pix2pix_model_epoch_1.pth --list_file val_list.txt --batch_size 4 --num_workers 0 --output_dir eval_results
```

Quick verified evaluation command used in this assignment:

```bash
python evaluate.py --checkpoint checkpoints/pix2pix_model_epoch_1.pth --list_file "D:/HW/DigitalImageProcessing/DIP-Teaching/Assignments/02_DIPwithPyTorch/Pix2Pix/val_list.txt" --batch_size 4 --num_workers 0 --max_steps 1 --output_dir eval_results
```

Verified evaluation output:

```text
Eval Step [1/25], Loss: 0.7876
Average L1 Loss: 0.7876
```

The evaluation script:

- loads a checkpoint
- runs inference on the validation set
- computes average L1 loss
- saves side-by-side input/target/output comparisons to `eval_results/`

## Pre-trained Models

A small trained checkpoint is included for reproducibility and demonstration:

- `Pix2Pix/checkpoints/pix2pix_model_epoch_1.pth`

Details:

- Dataset: Facades
- Training length: 1 epoch smoke test
- Batch size: 4
- Train steps used: 2
- Validation steps used: 1

This checkpoint is not a fully converged model. Its purpose is to provide:

- a loadable weight file for the template
- a working example for `evaluate.py`
- evidence that the train/evaluate pipeline runs correctly

## Results

### Poisson Image Editing

Expected observations:

| Example | Description | Observation |
| --- | --- | --- |
| Example 1 | Select a foreground object and place it near the center of the background | The object blends more naturally than direct pasting |
| Example 2 | Select a local region and paste it to a texture-different area | The result is still feasible, but the boundary may be slightly less smooth |

Analysis:

- Mask quality strongly affects blending quality.
- Gradient-based blending is more natural than naive copy-paste.
- Large appearance differences between source and target regions make blending harder.

### Pix2Pix

Smoke-test training result:

```text
Epoch [1/1], Step [1/100], Loss: 0.8290
Epoch [1/1], Step [2/100], Loss: 0.7796
Epoch [1/1], Validation Loss: 0.7876
```

Summary table:

| Model | Dataset | Epochs | Train Loss | Val Loss | Notes |
| --- | --- | --- | --- | --- | --- |
| FCN encoder-decoder | Facades | 1 | 0.7796 (last shown) | 0.7876 | pipeline verified |

Generated result folders:

- `Pix2Pix/checkpoints/`
- `Pix2Pix/eval_results/`

Conclusion:

- The FCN encoder-decoder can run forward and backward successfully.
- The training and evaluation pipeline is complete.
- Better image quality requires longer training and probably a larger dataset.

## Contributing

This project is a course assignment submission rather than a collaborative open-source project.

Possible future improvements:

- add skip connections to build a U-Net style generator
- add adversarial loss in addition to L1 loss
- train on a larger Pix2Pix dataset
- tune batch size, learning rate, and training epochs
- further improve Poisson blending efficiency

## Files included in submission

- `README.md`
- `run_blending_gradio.py`
- `Pix2Pix/FCN_network.py`
- `Pix2Pix/train.py`
- `Pix2Pix/evaluate.py`
- `Pix2Pix/facades_dataset.py`
- `Pix2Pix/download_facades_dataset.py`
- `Pix2Pix/checkpoints/pix2pix_model_epoch_1.pth`

This README follows the required template structure:

- `Requirements`
- `Training`
- `Evaluation`
- `Pre-trained Models`
- `Results`
- `Contributing`
