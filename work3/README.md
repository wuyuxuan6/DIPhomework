# Assignment 3 Report - Bundle Adjustment

## 基本信息

- 姓名：吴宇轩
- 学号：SA25001066
- 日期：2026-04-30

## 实验环境

- 操作系统：Windows
- Python：3.10
- Conda 环境：`ba_env`
- 主要依赖：
  - `numpy`
  - `matplotlib`
  - `opencv-python`
  - `torch==2.11.0+cpu`
  - `colmap`

## Task 1：使用 PyTorch 实现 Bundle Adjustment

### 实验目标

本实验根据 50 个视角下的 2D 观测点，联合优化共享焦距、每个视角的相机外参以及 20000 个三维点坐标，从而恢复场景的三维结构。

### 实现说明

实现代码位于 [bundle_adjustment.py](/D:/HW/DigitalImageProcessing/DIP-Teaching/Assignments/03_BundleAdjustment/bundle_adjustment.py:1)。

本实现的主要思路如下：

- 使用 XYZ Euler 角参数化相机旋转矩阵。
- 将所有视角的焦距设置为共享参数并参与联合优化。
- 按照题目给定的坐标系和投影公式实现重投影：
  - `u = -f * Xc / Zc + cx`
  - `v =  f * Yc / Zc + cy`
- 将相机平移初始化在 `[0, 0, -2.5]` 附近。
- 先根据初始相机参数进行多视图三角化，得到初始三维点。
- 采用两阶段优化策略：
  - 第一阶段仅优化焦距和相机参数；
  - 第二阶段联合优化焦距、相机参数和三维点坐标。

### 运行命令

```powershell
conda activate ba_env
cd D:\HW\DigitalImageProcessing\DIP-Teaching\Assignments\03_BundleAdjustment
python bundle_adjustment.py --camera_only_steps 200 --joint_steps 800 --output_dir outputs
```

### 实验结果

- 优化后焦距：`1346.1478`
- 重建三维点数量：`20000`
- 记录的优化迭代次数：`1000`
- 最终重投影损失：`211.1869`

生成文件如下：

- [outputs/ba_result.npz](/D:/HW/DigitalImageProcessing/DIP-Teaching/Assignments/03_BundleAdjustment/outputs/ba_result.npz)
- [outputs/reconstructed_points.obj](/D:/HW/DigitalImageProcessing/DIP-Teaching/Assignments/03_BundleAdjustment/outputs/reconstructed_points.obj)
- [outputs/loss_curve.png](/D:/HW/DigitalImageProcessing/DIP-Teaching/Assignments/03_BundleAdjustment/outputs/loss_curve.png)
- [outputs/reprojection_vis](/D:/HW/DigitalImageProcessing/DIP-Teaching/Assignments/03_BundleAdjustment/outputs/reprojection_vis)

### 结果分析

优化过程中损失函数持续下降，说明估计得到的相机参数和三维点坐标逐渐与观测到的二维投影保持一致。最终导出的彩色点云文件可以使用 MeshLab 等三维软件进一步进行可视化查看。

## Task 2：使用 COLMAP 进行三维重建

### 实验目标

本实验使用 COLMAP 命令行工具，对 `data/images/` 中的 50 张多视图图像执行完整的稀疏重建与稠密重建流程。

### 运行命令

```powershell
conda activate ba_env
cd D:\HW\DigitalImageProcessing\DIP-Teaching\Assignments\03_BundleAdjustment

colmap feature_extractor --database_path data\colmap\database.db --image_path data\images --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1
colmap exhaustive_matcher --database_path data\colmap\database.db
colmap mapper --database_path data\colmap\database.db --image_path data\images --output_path data\colmap\sparse
colmap image_undistorter --image_path data\images --input_path data\colmap\sparse\0 --output_path data\colmap\dense
colmap patch_match_stereo --workspace_path data\colmap\dense
colmap stereo_fusion --workspace_path data\colmap\dense --output_path data\colmap\dense\fused.ply
```

### 实验结果

- 稀疏重建：
  - 注册图像数量：`50`
  - 稀疏三维点数量：`1709`
- 稠密重建：
  - 融合点云顶点数量：`113080`

生成文件如下：

- [data/colmap/database.db](/D:/HW/DigitalImageProcessing/DIP-Teaching/Assignments/03_BundleAdjustment/data/colmap/database.db)
- [data/colmap/sparse/0](/D:/HW/DigitalImageProcessing/DIP-Teaching/Assignments/03_BundleAdjustment/data/colmap/sparse/0)
- [data/colmap/dense/fused.ply](/D:/HW/DigitalImageProcessing/DIP-Teaching/Assignments/03_BundleAdjustment/data/colmap/dense/fused.ply)

### 结果分析

稀疏重建阶段成功完成了全部 50 张图像的注册，说明图像之间具备较好的视角重叠关系。稠密重建进一步恢复了更加细致的表面几何结构，得到的点云密度明显高于稀疏重建结果。

## 提交文件

- [README.md](/D:/HW/DigitalImageProcessing/DIP-Teaching/Assignments/03_BundleAdjustment/README.md)
- [bundle_adjustment.py](/D:/HW/DigitalImageProcessing/DIP-Teaching/Assignments/03_BundleAdjustment/bundle_adjustment.py:1)
- [run_colmap.sh](/D:/HW/DigitalImageProcessing/DIP-Teaching/Assignments/03_BundleAdjustment/run_colmap.sh:1)
- [report.md](/D:/HW/DigitalImageProcessing/DIP-Teaching/Assignments/03_BundleAdjustment/report.md:1)
- [environment.yml](/D:/HW/DigitalImageProcessing/DIP-Teaching/Assignments/03_BundleAdjustment/environment.yml:1)
- [outputs](/D:/HW/DigitalImageProcessing/DIP-Teaching/Assignments/03_BundleAdjustment/outputs)
- [data/colmap](/D:/HW/DigitalImageProcessing/DIP-Teaching/Assignments/03_BundleAdjustment/data/colmap)
