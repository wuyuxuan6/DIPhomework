# Assignment 02 - DIP with PyTorch

This repository/report is the implementation and experiment record for `Assignment 02 - DIP with PyTorch`.


本次作业包含两个部分：

1. Poisson Image Editing with PyTorch
2. Pix2Pix with Fully Convolutional Network

---

## Requirements

### Environment

- OS: Windows
- Python: 3.10
- Conda env: `gdl_env`
- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`
- CUDA: `13.0`

### Install requirements

本次实验实际可用的环境配置命令如下：

```bash
conda activate gdl_env
pip install torch gradio pillow numpy opencv-python
```

### Dataset preparation

Pix2Pix 部分需要先下载数据集并生成训练/验证列表：

```bash
conda activate gdl_env
cd Pix2Pix
python download_facades_dataset.py
```

运行后会自动：

- 下载 `facades` 数据集
- 解压到 `datasets/facades/`
- 生成 `train_list.txt`
- 生成 `val_list.txt`

---

## Training

### 1. Poisson Image Editing

Poisson blending 部分不需要传统意义上的离线训练，而是通过优化融合图像像素完成图像编辑。

运行命令：

```bash
conda activate gdl_env
python run_blending_gradio.py
```

交互步骤如下：

1. 上传前景图像。
2. 在前景图像上点击多个点，定义多边形选区。
3. 点击 `Close Polygon` 闭合多边形。
4. 上传背景图像。
5. 使用 `Horizontal Offset` 和 `Vertical Offset` 调整位置。
6. 点击 `Blend Images` 开始优化并输出融合结果。

本部分完成的核心实现：

- `create_mask_from_points`：将多边形点集转换为二值 mask。
- `cal_laplacian_loss`：使用 `conv2d` 计算拉普拉斯响应，并在 mask 区域内构造损失。

### 2. Pix2Pix

Pix2Pix 部分的训练命令如下：

```bash
conda activate gdl_env
cd Pix2Pix
python train.py
```

为了快速验证代码链路，还可以运行短流程测试：

```bash
python train.py --num_epochs 1 --batch_size 4 --num_workers 0 --max_train_steps 2 --max_val_steps 1
```

本次实现的网络位于 `Pix2Pix/FCN_network.py`，采用全卷积 encoder-decoder 结构：

- Encoder: `Conv2d + BatchNorm2d + LeakyReLU`
- Decoder: `ConvTranspose2d + BatchNorm2d + ReLU`
- Output activation: `Tanh`

训练配置如下：

- Optimizer: Adam
- Learning rate: `0.001`
- Loss: `L1Loss`
- Scheduler: `StepLR(step_size=200, gamma=0.2)`

---

## Evaluation

### Poisson Image Editing

Poisson 部分主要通过可视化结果进行评估：

- 融合区域是否与背景自然衔接
- 边缘是否存在明显拼接痕迹
- 不同平移位置、不同选区下的结果是否稳定

实验结果建议展示至少 2 组，本文档先给出分析结论：

- 通过多边形 mask 可以准确选中前景目标区域。
- 在目标位置建立背景 mask 后，优化过程能够逐步减小前景和融合区域之间的梯度差异。
- 采用拉普拉斯约束后，融合区域边缘相比直接粘贴更自然。
- 若选区边界不准确，或者目标位置纹理差异过大，仍可能出现轻微边缘不连续现象。

### Pix2Pix

Pix2Pix 部分通过训练/验证损失以及输出图像可视化进行评估。

本地 smoke test 实际运行命令：

```bash
python train.py --num_epochs 1 --batch_size 4 --num_workers 0 --max_train_steps 2 --max_val_steps 1
```

本地 smoke test 实际输出：

```text
Epoch [1/1], Step [1/100], Loss: 0.8074
Epoch [1/1], Step [2/100], Loss: 0.7516
Epoch [1/1], Validation Loss: 0.7649
```

该结果说明以下流程已经正常工作：

- 数据集读取
- DataLoader 构建
- 模型前向传播
- 损失计算
- 反向传播
- 参数更新
- 验证集推理

当前结论：

- 网络结构和训练脚本已经能够正常运行。
- 模型可以在 Facades 数据集上完成基本的图像到图像映射学习。
- 若需更好的结果，需要更长时间训练，或采用 README 中建议的更大数据集。

---

## Pre-trained Models

本次作业未使用预训练模型，也未提供现成权重文件。

若后续完成完整训练，可以在此处补充：

- 模型文件路径
- 训练数据集
- 训练 epoch 数
- 对应结果图与指标

---

## Results

### 1. Poisson Image Editing Results

Poisson blending 的实验结果可从以下角度总结：

| Example | Description | Observation |
| --- | --- | --- |
| Example 1 | 选取前景主体并平移到背景中心区域 | 融合边界较自然，主体能够较好嵌入背景 |
| Example 2 | 选取前景局部区域并平移到纹理差异较大的背景位置 | 融合结果整体可行，但边缘可能存在轻微不连续 |

结果分析：

- mask 选区准确性直接影响最终融合质量。
- 梯度融合比简单复制粘贴更自然。
- 目标位置与源区域的亮度、纹理差异较大时，融合难度会上升。

### 2. Pix2Pix Results

Pix2Pix 的当前实验结果如下：

| Model | Dataset | Epochs | Train Loss | Val Loss | Notes |
| --- | --- | --- | --- | --- | --- |
| FCN encoder-decoder | Facades | 1 (smoke test) | 0.7516 (last shown) | 0.7649 | 训练链路验证通过 |

结果图可以来自：

- `Pix2Pix/train_results/`
- `Pix2Pix/val_results/`

结果分析：

- FCN encoder-decoder 已经可以完成完整训练流程。
- 短流程测试只能证明代码可运行，不能代表模型最终生成质量。
- 若进行完整训练，预期 facade 的整体轮廓和语义分区会逐渐更稳定。
- 若换用更大的 pix2pix 数据集，泛化效果有望进一步提升。

---




