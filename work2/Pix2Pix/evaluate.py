import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from FCN_network import FullyConvNetwork
from facades_dataset import FacadesDataset


def tensor_to_image(tensor):
    image = tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image + 1) / 2
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)


def save_results(inputs, targets, outputs, output_dir, batch_index):
    os.makedirs(output_dir, exist_ok=True)
    num_images = min(inputs.shape[0], targets.shape[0], outputs.shape[0])
    for i in range(num_images):
        input_img = tensor_to_image(inputs[i])
        target_img = tensor_to_image(targets[i])
        output_img = tensor_to_image(outputs[i])
        comparison = np.hstack((input_img, target_img, output_img))
        cv2.imwrite(
            os.path.join(output_dir, f"batch_{batch_index + 1}_sample_{i + 1}.png"),
            comparison,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Pix2Pix FCN model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--list_file", type=str, default="val_list.txt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = FacadesDataset(list_file=args.list_file)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = FullyConvNetwork().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.L1Loss()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for step, (image_rgb, image_semantic) in enumerate(dataloader):
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            outputs = model(image_rgb)
            loss = criterion(outputs, image_semantic)
            total_loss += loss.item()
            total_steps += 1

            save_results(image_rgb, image_semantic, outputs, args.output_dir, step)
            print(f"Eval Step [{step + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

            if args.max_steps is not None and total_steps >= args.max_steps:
                break

    avg_loss = total_loss / max(total_steps, 1)
    print(f"Average L1 Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
