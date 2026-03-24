import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=2.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """
    if image is None:
        return None

    image = np.asarray(image)
    warped_image = np.array(image)

    num_pairs = min(len(source_pts), len(target_pts))
    if num_pairs == 0:
        return warped_image

    source_pts = np.asarray(source_pts[:num_pairs], dtype=np.float32)
    target_pts = np.asarray(target_pts[:num_pairs], dtype=np.float32)

    height, width = image.shape[:2]

    # Backward warping: each pixel on the output image is mapped back to
    # the source image. To make source control points move to target
    # control points, we evaluate the inverse MLS map: target -> source.
    p = target_pts
    q = source_pts

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    pixels = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

    diff = pixels[:, None, :] - p[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)

    exact_mask = dist2 < eps
    weights = 1.0 / np.power(dist2 + eps, alpha)
    weights_sum = np.sum(weights, axis=1, keepdims=True)

    p_star = (weights[:, :, None] * p[None, :, :]).sum(axis=1) / weights_sum
    q_star = (weights[:, :, None] * q[None, :, :]).sum(axis=1) / weights_sum

    if num_pairs == 1:
        mapped = (pixels + (q[0] - p[0])).astype(np.float32)
    else:
        p_hat = p[None, :, :] - p_star[:, None, :]
        q_hat = q[None, :, :] - q_star[:, None, :]
        v_hat = pixels - p_star

        mu = np.sum(weights * np.sum(p_hat * p_hat, axis=2), axis=1, keepdims=True)
        mu = np.maximum(mu, eps)

        real_part = np.sum(weights * np.sum(q_hat * p_hat, axis=2), axis=1, keepdims=True)
        imag_part = np.sum(
            weights * (q_hat[:, :, 1] * p_hat[:, :, 0] - q_hat[:, :, 0] * p_hat[:, :, 1]),
            axis=1,
            keepdims=True,
        )

        mapped = np.empty_like(v_hat, dtype=np.float32)
        mapped[:, 0] = (
            (real_part[:, 0] * v_hat[:, 0] - imag_part[:, 0] * v_hat[:, 1]) / mu[:, 0]
            + q_star[:, 0]
        )
        mapped[:, 1] = (
            (imag_part[:, 0] * v_hat[:, 0] + real_part[:, 0] * v_hat[:, 1]) / mu[:, 0]
            + q_star[:, 1]
        )

    # Control points should map exactly, which also avoids numerical noise
    # near singular locations.
    if np.any(exact_mask):
        exact_rows, exact_cols = np.where(exact_mask)
        mapped[exact_rows] = q[exact_cols]

    map_x = np.clip(mapped[:, 0], 0, width - 1).reshape(height, width)
    map_y = np.clip(mapped[:, 1], 0, height - 1).reshape(height, width)

    warped_image = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
