# Assignment 02 - DIP with PyTorch

本实验完成了 `Assignment 02 - DIP with PyTorch` 的两个部分：

1. 基于 PyTorch 的 Poisson Image Editing
2. 基于全卷积网络的 Pix2Pix 图像到图像翻译

## Requirements

### Environment

- Operating System: Windows
- Python: 3.10
- Conda Environment: `gdl_env`
- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`
- CUDA: `13.0`

### Install requirements

```bash
conda activate gdl_env
pip install torch gradio pillow numpy opencv-python
```

### Dataset preparation

Pix2Pix 部分使用 Facades 数据集。数据下载与训练/验证列表生成命令如下：

```bash
conda activate gdl_env
cd Pix2Pix
python download_facades_dataset.py
```

运行该脚本后将自动完成以下工作：

- 下载 `facades` 数据集
- 解压到 `datasets/facades/`
- 生成 `train_list.txt`
- 生成 `val_list.txt`

## Training

### 1. Poisson Image Editing

Poisson blending 部分通过优化方式完成前景区域与背景区域的梯度融合，不涉及传统监督学习意义上的模型训练。

运行命令如下：

```bash
conda activate gdl_env
python run_blending_gradio.py
```

交互流程如下：

1. 上传前景图像。
2. 在前景图像上点击多个点，定义多边形区域。
3. 点击 `Close Polygon` 闭合选区。
4. 上传背景图像。
5. 使用 `Horizontal Offset` 与 `Vertical Offset` 调整粘贴位置。
6. 点击 `Blend Images` 执行优化，得到融合结果。

本部分实现的关键内容如下：

- `create_mask_from_points`：根据多边形点集生成二值掩码
- `cal_laplacian_loss`：计算前景区域与融合区域之间的拉普拉斯梯度损失

### 2. Pix2Pix

Pix2Pix 部分采用全卷积 encoder-decoder 结构进行图像到图像翻译。训练命令如下：

```bash
conda activate gdl_env
cd Pix2Pix
python train.py --train_list train_list.txt --val_list val_list.txt --output_dir checkpoints_full
```

网络结构如下：

- Encoder: `Conv2d + BatchNorm2d + LeakyReLU`
- Decoder: `ConvTranspose2d + BatchNorm2d + ReLU`
- Output Activation: `Tanh`

训练配置如下：

- Optimizer: Adam
- Learning Rate: `0.001`
- Loss Function: `L1Loss`
- Scheduler: `StepLR(step_size=200, gamma=0.2)`
- Epochs: `300`

本次完整训练使用 Facades 数据集完成 300 个 epoch，并在训练过程中每 50 个 epoch 保存一次权重。

## Evaluation

### 1. Poisson Image Editing

Poisson blending 的评估以可视化效果为主，重点观察以下内容：

- 融合区域与背景之间的边界是否自然
- 与直接复制粘贴相比，是否具有更平滑的过渡效果
- 在不同目标位置与不同选区条件下，融合结果是否稳定

### 2. Pix2Pix

Pix2Pix 部分提供了独立评估脚本 `evaluate.py`。该脚本的功能如下：

- 加载指定模型权重
- 读取验证集图像列表
- 对验证集执行前向推理
- 计算输出结果与目标图像之间的平均 `L1 Loss`
- 将输入图像、目标图像和生成结果按水平方向拼接后保存到 `eval_results/`

评估命令如下：

```bash
conda activate gdl_env
cd Pix2Pix
python evaluate.py --checkpoint checkpoints_full/pix2pix_model_epoch_300.pth --list_file val_list.txt --batch_size 4 --num_workers 0 --output_dir eval_results
```

在本实验中，还使用如下命令完成过一次快速验证：

```bash
python evaluate.py --checkpoint checkpoints/pix2pix_model_epoch_1.pth --list_file "D:/HW/DigitalImageProcessing/DIP-Teaching/Assignments/02_DIPwithPyTorch/Pix2Pix/val_list.txt" --batch_size 4 --num_workers 0 --max_steps 1 --output_dir eval_results
```

对应输出为：

```text
Eval Step [1/25], Loss: 0.7876
Average L1 Loss: 0.7876
```

## Pre-trained Models

本实验提交以下模型权重：

- `Pix2Pix/checkpoints_full/pix2pix_model_epoch_300.pth`

同时保留训练过程中保存的中间权重：

- `Pix2Pix/checkpoints_full/pix2pix_model_epoch_50.pth`
- `Pix2Pix/checkpoints_full/pix2pix_model_epoch_100.pth`
- `Pix2Pix/checkpoints_full/pix2pix_model_epoch_150.pth`
- `Pix2Pix/checkpoints_full/pix2pix_model_epoch_200.pth`
- `Pix2Pix/checkpoints_full/pix2pix_model_epoch_250.pth`
- `Pix2Pix/checkpoints_full/pix2pix_model_epoch_300.pth`

其中 `pix2pix_model_epoch_300.pth` 为本次完整训练结束后得到的最终模型。

## Results

### 1. Poisson Image Editing

Poisson Image Editing 的实验目标是将前景图像中的选定区域粘贴到背景图像中，并通过梯度约束减弱边界不连续现象。实验结果表明：

- 当前景区域与目标背景在亮度和纹理上差异较小时，融合边界更加自然
- 当目标背景纹理复杂或颜色差异较大时，边界附近仍可能出现轻微不连续
- 与直接复制粘贴相比，Poisson blending 在视觉过渡上具有更好的自然性

### 2. Pix2Pix

完整训练过程中，模型损失呈现出较明显的下降趋势。训练初期与结束阶段的部分结果如下：

```text
Epoch [1/300], Validation Loss: 0.7443
Epoch [50/300], Validation Loss: 0.3712
Epoch [100/300], Validation Loss: 0.3770
Epoch [150/300], Validation Loss: 0.3768
Epoch [200/300], Validation Loss: 0.3776
Epoch [250/300], Validation Loss: 0.3791
Epoch [300/300], Validation Loss: 0.3791
```

训练中较低的验证损失出现在前中期，约在第 35 个 epoch 左右达到 `0.3614`。在后续训练中，训练损失继续下降，但验证损失基本稳定在 `0.37` 到 `0.38` 区间，说明模型后期主要在训练集上进一步拟合，而验证集提升有限。

结果汇总如下：

| Model | Dataset | Epochs | Final Train Loss | Final Val Loss | Best Observed Val Loss |
| --- | --- | --- | --- | --- | --- |
| FCN encoder-decoder | Facades | 300 | 0.0443 | 0.3791 | 0.3614 |

训练和验证过程中的可视化结果保存在以下目录：

- `Pix2Pix/train_results/`
- `Pix2Pix/val_results/`
- `Pix2Pix/eval_results/`

从结果图可以观察到，模型能够学习到 facade 图像的整体布局与主要结构信息，但在细节边缘、局部纹理和小尺度区域上仍存在一定模糊现象。这与当前模型仅使用全卷积 encoder-decoder 结构、未引入 skip connection 和 adversarial loss 有关。

## Contributing

本项目为课程作业提交内容，不涉及协作开发流程。

后续可进一步改进的方向包括：

- 为生成器加入 skip connection，构建更接近 U-Net 的结构
- 在 `L1 Loss` 之外引入 adversarial loss
- 使用规模更大的 Pix2Pix 数据集
- 调整 batch size、学习率与训练轮数
- 进一步优化 Poisson blending 的求解效率

## Files included in submission

- `README.md`
- `run_blending_gradio.py`
- `Pix2Pix/FCN_network.py`
- `Pix2Pix/train.py`
- `Pix2Pix/evaluate.py`
- `Pix2Pix/facades_dataset.py`
- `Pix2Pix/download_facades_dataset.py`
- `Pix2Pix/checkpoints_full/pix2pix_model_epoch_50.pth`
- `Pix2Pix/checkpoints_full/pix2pix_model_epoch_100.pth`
- `Pix2Pix/checkpoints_full/pix2pix_model_epoch_150.pth`
- `Pix2Pix/checkpoints_full/pix2pix_model_epoch_200.pth`
- `Pix2Pix/checkpoints_full/pix2pix_model_epoch_250.pth`
- `Pix2Pix/checkpoints_full/pix2pix_model_epoch_300.pth`

本 README 按照要求组织为以下结构：

- `Requirements`
- `Training`
- `Evaluation`
- `Pre-trained Models`
- `Results`
- `Contributing`
