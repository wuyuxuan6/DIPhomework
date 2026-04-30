import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


IMAGE_SIZE = 1024
CX = IMAGE_SIZE / 2.0
CY = IMAGE_SIZE / 2.0


def load_data(data_dir):
    data_dir = Path(data_dir)
    points2d_npz = np.load(data_dir / "points2d.npz")
    view_keys = sorted(points2d_npz.files)
    observations = np.stack([points2d_npz[key][:, :2] for key in view_keys], axis=0).astype(np.float32)
    visibility = np.stack([points2d_npz[key][:, 2] for key in view_keys], axis=0).astype(bool)
    colors = np.load(data_dir / "points3d_colors.npy").astype(np.float32) / 255.0
    return view_keys, observations, visibility, colors


def euler_xyz_to_matrix(angles):
    x, y, z = angles.unbind(dim=-1)
    cx, cy, cz = torch.cos(x), torch.cos(y), torch.cos(z)
    sx, sy, sz = torch.sin(x), torch.sin(y), torch.sin(z)

    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)

    rx = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cx, -sx], dim=-1),
            torch.stack([zeros, sx, cx], dim=-1),
        ],
        dim=-2,
    )
    ry = torch.stack(
        [
            torch.stack([cy, zeros, sy], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sy, zeros, cy], dim=-1),
        ],
        dim=-2,
    )
    rz = torch.stack(
        [
            torch.stack([cz, -sz, zeros], dim=-1),
            torch.stack([sz, cz, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    return rz @ ry @ rx


def project_points(points3d, rotations, translations, focal):
    camera_points = torch.einsum("vij,nj->vni", rotations, points3d) + translations[:, None, :]
    z = camera_points[..., 2]
    safe_z = torch.where(z.abs() < 1e-6, torch.sign(z) * 1e-6, z)
    u = -focal * camera_points[..., 0] / safe_z + CX
    v = focal * camera_points[..., 1] / safe_z + CY
    return torch.stack([u, v], dim=-1), camera_points


def build_camera_matrices(rotations, translations, focal):
    num_views = rotations.shape[0]
    k = np.array([[-focal, 0.0, CX], [0.0, focal, CY], [0.0, 0.0, 1.0]], dtype=np.float64)
    projection_mats = []
    for i in range(num_views):
        extrinsic = np.concatenate([rotations[i], translations[i, :, None]], axis=1)
        projection_mats.append(k @ extrinsic)
    return projection_mats


def triangulate_points(observations, visibility, rotations, translations, focal):
    projection_mats = build_camera_matrices(rotations, translations, focal)
    num_points = observations.shape[1]
    points3d = np.zeros((num_points, 3), dtype=np.float32)

    for point_idx in range(num_points):
        visible_views = np.where(visibility[:, point_idx])[0]
        if len(visible_views) < 2:
            continue

        rows = []
        for view_idx in visible_views:
            u, v = observations[view_idx, point_idx]
            p = projection_mats[view_idx]
            rows.append(u * p[2] - p[0])
            rows.append(v * p[2] - p[1])
        a = np.stack(rows, axis=0)
        _, _, vh = np.linalg.svd(a, full_matrices=False)
        homog = vh[-1]
        points3d[point_idx] = (homog[:3] / homog[3]).astype(np.float32)

    return points3d


def save_colored_obj(path, points3d, colors):
    with open(path, "w", encoding="utf-8") as f:
        for point, color in zip(points3d, colors):
            f.write(
                f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
            )


def plot_loss_curve(loss_history, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Reprojection MSE")
    plt.title("Bundle Adjustment Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_reprojection_visualization(data_dir, output_dir, view_keys, observations, visibility, predicted):
    vis_dir = Path(output_dir) / "reprojection_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    selected = [0, 12, 25, 37, 49]
    for idx in selected:
        key = view_keys[idx]
        img = cv2.imread(str(data_dir / "images" / f"{key}.png"))
        obs = observations[idx]
        pred = predicted[idx]
        vis = visibility[idx]

        for j in np.where(vis)[0]:
            ox, oy = int(round(obs[j, 0])), int(round(obs[j, 1]))
            px, py = int(round(pred[j, 0])), int(round(pred[j, 1]))
            cv2.circle(img, (ox, oy), 2, (0, 255, 0), -1)
            cv2.circle(img, (px, py), 2, (0, 0, 255), -1)
        cv2.imwrite(str(vis_dir / f"{key}_reprojection.png"), img)


def optimize_bundle_adjustment(
    observations,
    visibility,
    colors,
    output_dir,
    init_distance=2.5,
    init_focal=886.81,
    camera_only_steps=200,
    joint_steps=400,
):
    device = torch.device("cpu")
    obs_t = torch.from_numpy(observations).to(device)
    vis_t = torch.from_numpy(visibility).to(device)

    num_views = observations.shape[0]
    yaw_init = torch.linspace(-70.0, 70.0, num_views, dtype=torch.float32) * math.pi / 180.0
    euler_init = torch.zeros((num_views, 3), dtype=torch.float32)
    euler_init[:, 1] = yaw_init
    translation_init = torch.zeros((num_views, 3), dtype=torch.float32)
    translation_init[:, 2] = -init_distance

    init_rot = euler_xyz_to_matrix(euler_init).numpy()
    init_points = triangulate_points(
        observations,
        visibility,
        init_rot,
        translation_init.numpy(),
        init_focal,
    )

    points3d = torch.nn.Parameter(torch.from_numpy(init_points).to(device))
    euler_angles = torch.nn.Parameter(euler_init.to(device))
    translations = torch.nn.Parameter(translation_init.to(device))
    log_focal = torch.nn.Parameter(torch.tensor(math.log(init_focal), dtype=torch.float32, device=device))

    def current_focal():
        return torch.exp(log_focal)

    loss_history = []

    camera_optimizer = torch.optim.Adam(
        [
            {"params": [euler_angles], "lr": 3e-3},
            {"params": [translations], "lr": 3e-3},
            {"params": [log_focal], "lr": 1e-3},
        ]
    )

    for step in range(camera_only_steps):
        rotations = euler_xyz_to_matrix(euler_angles)
        pred_2d, _ = project_points(points3d.detach(), rotations, translations, current_focal())
        diff = pred_2d - obs_t
        loss = (diff.square().sum(dim=-1)[vis_t]).mean()

        camera_optimizer.zero_grad()
        loss.backward()
        camera_optimizer.step()
        loss_history.append(loss.item())

        if step % 20 == 0 or step == camera_only_steps - 1:
            print(f"[Camera Warmup] step {step + 1}/{camera_only_steps}, loss={loss.item():.6f}, f={current_focal().item():.3f}")

    joint_optimizer = torch.optim.Adam(
        [
            {"params": [points3d], "lr": 5e-3},
            {"params": [euler_angles], "lr": 1e-3},
            {"params": [translations], "lr": 1e-3},
            {"params": [log_focal], "lr": 5e-4},
        ]
    )

    for step in range(joint_steps):
        rotations = euler_xyz_to_matrix(euler_angles)
        pred_2d, camera_points = project_points(points3d, rotations, translations, current_focal())
        diff = pred_2d - obs_t
        reproj_loss = (diff.square().sum(dim=-1)[vis_t]).mean()
        z_penalty = torch.relu(camera_points[..., 2]).square().mean()
        loss = reproj_loss + 1e-2 * z_penalty

        joint_optimizer.zero_grad()
        loss.backward()
        joint_optimizer.step()
        loss_history.append(reproj_loss.item())

        if step % 20 == 0 or step == joint_steps - 1:
            print(
                f"[Joint BA] step {step + 1}/{joint_steps}, reproj_loss={reproj_loss.item():.6f}, "
                f"z_penalty={z_penalty.item():.6f}, f={current_focal().item():.3f}"
            )

    rotations = euler_xyz_to_matrix(euler_angles)
    pred_2d, _ = project_points(points3d, rotations, translations, current_focal())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "ba_result.npz",
        points3d=points3d.detach().cpu().numpy(),
        euler_angles=euler_angles.detach().cpu().numpy(),
        translations=translations.detach().cpu().numpy(),
        focal=np.array([current_focal().item()], dtype=np.float32),
        loss_history=np.array(loss_history, dtype=np.float32),
    )
    save_colored_obj(output_dir / "reconstructed_points.obj", points3d.detach().cpu().numpy(), colors)
    plot_loss_curve(loss_history, output_dir / "loss_curve.png")

    return pred_2d.detach().cpu().numpy(), loss_history, current_focal().item()


def parse_args():
    parser = argparse.ArgumentParser(description="Bundle Adjustment from scratch with PyTorch.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--init_distance", type=float, default=2.5)
    parser.add_argument("--init_focal", type=float, default=886.81)
    parser.add_argument("--camera_only_steps", type=int, default=200)
    parser.add_argument("--joint_steps", type=int, default=400)
    return parser.parse_args()


def main():
    args = parse_args()
    view_keys, observations, visibility, colors = load_data(args.data_dir)
    predicted, loss_history, final_focal = optimize_bundle_adjustment(
        observations,
        visibility,
        colors,
        args.output_dir,
        init_distance=args.init_distance,
        init_focal=args.init_focal,
        camera_only_steps=args.camera_only_steps,
        joint_steps=args.joint_steps,
    )
    save_reprojection_visualization(args.data_dir, args.output_dir, view_keys, observations, visibility, predicted)
    print(f"Optimization finished. Final focal length: {final_focal:.4f}")
    print(f"Final reprojection loss: {loss_history[-1]:.6f}")
    print(f"Saved results to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
