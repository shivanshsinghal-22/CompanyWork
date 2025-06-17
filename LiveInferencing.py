import cv2
import numpy as np
import time
import threading
import queue
import math
from ultralytics import YOLO
from PIL import Image, ImageTk
import tkinter as tk
import gc
import torch # Assuming torch is needed for DEVICE, though not used in the provided functions
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from segmentation_models_pytorch import DeepLabV3Plus
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import concurrent.futures
from rembg import remove, new_session
from PIL import Image
import onnxruntime as ort

model_weight_path = "/kaggle/input/hairmodel/pytorch/default/1/deeplabv3_efficientnetb4_epoch200.pth"
grid_rows = 3
grid_cols = 3
patch_size = 1024
encoder_name = "timm-efficientnet-b4"
model = DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
model.load_state_dict(torch.load(model_weight_path, map_location="cuda"))
model = model.to("cuda")
model.eval()

# dummy_input = torch.randn(1, 3, 1024, 1024, ).to("cuda")

# onnx_path = "/kaggle/working/deeplabv3plus.onnx"

# # Export to ONNX
# torch.onnx.export(
#     model, 
#     dummy_input, 
#     onnx_path,
#     export_params=True,
#     opset_version=11,
#     do_constant_folding=True,
#     input_names=["input"],
#     output_names=["output"],
#     dynamic_axes={
#         'input': {0: 'batch_size', 2: 'height', 3: 'width'},
#         'output': {0: 'batch_size', 2: 'height', 3: 'width'}
#     }
# )

# del model, dummy_input
# gc.collect()
# torch.cuda.empty_cache()

# # Load ONNX session once globally
# onnx_session = ort.InferenceSession("/kaggle/working/deeplabv3plus.onnx", providers=["CUDAExecutionProvider"])

session = new_session("u2netp")
TEMPLATE_BIG = "/kaggle/input/reference-images/IMG_1371_nobg.png"
TEMPLATE_MINI = "/kaggle/input/reference-images/IMG_0590.png"
# Threshold constants
MAX_ANGLE = 15        # degrees (left-right angle difference)
MAX_KINK = 20         # degrees (absolute change between consecutive angles)
MAX_FOLD = -10        # degrees (minimum allowable signed change)
MAX_DENTS = 250       # maximum allowable base dents

# === Configuration ===
VIDEO_PATH = Path("/kaggle/input/videofile/GX010017.MP4")
IMAGE_FOLDER_PATH = Path("/kaggle/input/good-naans/PNG_images_good_naans")
FRAME_INTERVAL_SEC = 0.1
USE_GPU = True
DEVICE = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")

# Thread-safe, time ordered queue
pq = queue.Queue()          # (timestamp, ndarray)
global_executer = ThreadPoolExecutor(max_workers=5)
model = YOLO("yolov8l.pt").to("cuda")
predefined_bbox = (0, 0, 2560, 1440)
# Sentinel object used to tell worker to stop
_SENTINEL = object()

t = time.time()
@dataclass
class FrameProfile:
    t_enqueue: float = 0.0
    t_dequeue: float = 0.0
    t_shape: float   = 0.0
    t_cook: float    = 0.0
    t_hair: float    = 0.0

def crop_yolo(frame):
    results = model.predict(frame, device="cuda", verbose=False)
    for result in results:
        for box in result.boxes:
            if int(box.cls) in [53, 55] and box.conf >= 0.75:
                xmin, ymin, xmax, ymax = box.xyxy.tolist()[0]
                x1, y1, x2, y2 = predefined_bbox
                if xmin > x1 + 10 and ymin > y1 + 10 and xmax < x2 - 10 and ymax < y2 - 10:
                    return frame[int(ymin):int(ymax), int(xmin):int(xmax)]
    return None

def reader_thread():
    """Read frames and enqueue."""

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open video")
        pq.put(_SENTINEL)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame is not None:
            crop = crop_yolo(frame)
            if crop is not None:
                frame_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                profile = FrameProfile(t_enqueue=time.time() - t)
                pq.put((profile, frame_rgb))
                time.sleep(FRAME_INTERVAL_SEC)

    cap.release()
    pq.put(_SENTINEL)


# def reader_thread():

#     """Read images from a folder and enqueue them."""
#     if not IMAGE_FOLDER_PATH.is_dir():
#         print(f"Error: Image folder not found at {IMAGE_FOLDER_PATH}")
#         pq.put((time.time() - t, float('inf'), None, _SENTINEL))
#         return

#     # Get all image files (e.g., .png, .jpg, .jpeg)
#     image_files = sorted(list(IMAGE_FOLDER_PATH.glob("*.png")) +
#                          list(IMAGE_FOLDER_PATH.glob("*.jpg")) +
#                          list(IMAGE_FOLDER_PATH.glob("*.jpeg")))

#     if not image_files:
#         print(f"No image files found in {IMAGE_FOLDER_PATH}")
#         pq.put((time.time() - t, float('inf'), None, _SENTINEL))
#         return

#     for idx, image_path in enumerate(image_files):
#         frame = cv2.imread(str(image_path))
#         if frame is None:
#             print(f"Warning: Could not read {image_path}"); continue

#         profile = FrameProfile(idx=idx)
#         profile.t_enqueue = time.time() - t
#         pq.put((profile.t_enqueue, idx, profile, frame))  
#         time.sleep(FRAME_INTERVAL_SEC)
                                                     
#     # at EOF
#     pq.put((time.time() - t, float('inf'), None, _SENTINEL))

def worker_allocator():
    while True:
        item = pq.get()

        if item is _SENTINEL:
            print("Shutdown signal received. Shutting down workAllocator...")
            pq.put(_SENTINEL)  # In case there are other consumers
            break

        prof, frame = item
        global_executer.submit(dispatcher_thread, prof, frame)

    global_executer.shutdown(wait=True)
    cv2.destroyAllWindows()  # Close all OpenCV GUI windows on shutdown

        
def dispatcher_thread(prof, frame):
    prof.t_dequeue = time.time() - t

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        future_to_key = {
            pool.submit(process_single_image, frame): "shape",
            pool.submit(detect_overcooked_spots, frame): "overcooked",
            pool.submit(detect_hair, frame): "hair"
        }

        results = {}

        for fut in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[fut]
            try:
                result = fut.result()
                results[key] = result
                now = time.time() - t

                if key == "shape":
                    prof.t_shape = now
                elif key == "overcooked":
                    prof.t_cook = now
                elif key == "hair":
                    prof.t_hair = now

            except concurrent.futures.CancelledError:
                print(f"Task for {key} was cancelled.")
            except concurrent.futures.TimeoutError:
                print(f"Task for {key} timed out.")
            except Exception as e:
                print(f"Error processing {key}: {e}")

        print(f"{prof.idx} : (Absolute times: Enqueued: {prof.t_enqueue:.4f}, Dequeued: {prof.t_dequeue:.4f}, "
              f"Shape End: {prof.t_shape - prof.t_dequeue:.4f}, "
              f"Cook End: {prof.t_cook - prof.t_dequeue:.4f}, "
              f"Hair End: {prof.t_hair - prof.t_dequeue:.4f})")

        # Show results in 3 GUI windows
        if "shape" in results:
            cv2.imshow('Shape Detection', results["shape"]["annotated_image"])
        if "overcooked" in results:
            cv2.imshow('Overcooked Detection', results["overcooked"]["annotated_image"])
        if "hair" in results:
            cv2.imshow('Hair Detection', results["hair"]["annotated_image"])

        # Required to refresh OpenCV windows
        cv2.waitKey(1)


def detect_overcooked_spots(image):
    if image is None:
        return ValueError("Input Image is None")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 0), (43, 160, 90))

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    is_overcooked = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 60:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if 14 <= w <= 300 and 14 <= h <= 300:
            is_overcooked = True
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(image, "Spot", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return {
        "Result":is_overcooked,
        "annotated_image": image
    }

def detect_hair(
    frame_np: np.ndarray):
    """
    Detect hair in a NumPy frame using patch-wise segmentation with threading.
    Returns ("in-memory", "hair"/"okk") — no file is saved.
    """

    # global model_weight_path, grid_rows, grid_cols, patch_size, encoder_name
    # Convert NumPy frame to PIL
    image_pil = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)).convert("RGB")
    w, h = image_pil.size
    step_h = h // grid_rows
    step_w = w // grid_cols

    # Setup
    transform = transforms.Compose([transforms.ToTensor()])
    full_mask = np.zeros((patch_size * grid_rows, patch_size * grid_cols), dtype=np.uint8)
    lock = threading.Lock()

    def process_patch(i, j):
        
        nonlocal full_mask
        patch = image_pil.crop((j * step_w, i * step_h, (j+1) * step_w, (i+1) * step_h))
        patch = patch.resize((patch_size, patch_size), resample=Image.BILINEAR)
        patch_tensor = transform(patch).unsqueeze(0).to("cuda")

        with torch.no_grad():
            output = model(patch_tensor)
            mask = torch.sigmoid(output).squeeze().cpu().numpy()
            patch_mask_bin = (mask > 0.5).astype(np.uint8) * 255

        full_mask[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patch_mask_bin

        del patch_tensor, output
        torch.cuda.empty_cache()

    # Launch threads
    threads = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            t = threading.Thread(target=process_patch, args=(i, j))
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    result = True if (full_mask > 0).sum() > 0 else False
    return {
        "Result":result,
        "annotated_image":full_mask
    }

def process_single_image(image_input):

    def remove_bg(image_bgr):
        
        new_size = (640, 480)  
        resized_bgr = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_LANCZOS4)

        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        output_pil = remove(pil_img, session=session)
        output_rgb = np.array(output_pil)
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

        return output_bgr

    def extract_boundary(image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.flip(image, -1)

        if image is None:
            raise ValueError(f"Image at path '{image_path}' could not be loaded.")

        _, thresholded = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        boundary = cv2.morphologyEx(thresholded, cv2.MORPH_GRADIENT, kernel)

        return image, thresholded, boundary

    def _load_and_prep_template(template_path):

        template_full = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)

        if template_full is None:
            raise FileNotFoundError(f"Template image '{template_path}' not found.")

        if template_full.shape[0] < 20 or template_full.shape[1] < 20:
            raise ValueError(f"Template image '{template_path}' seems blank or too small.")

        h_full, w_full = template_full.shape[:2]
        if h_full > 1.5 * w_full and h_full > 200:
            template_img = template_full[h_full // 2:, :, :]
        else:
            template_img = template_full

        if len(template_img.shape) == 3 and template_img.shape[2] == 4:
            alpha = template_img[:,:,3] / 255.0
            background = np.ones_like(template_img[:,:,:3]) * 255
            bgr = (1.0 - alpha[:,:,None]) * background + alpha[:,:,None] * template_img[:,:,:3]
            image_gray = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            image = cv2.bitwise_not(image_gray)
        elif len(template_img.shape) == 3:
            image = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        else:
            image = template_img

        _, template_mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

        if np.sum(template_mask) < 10:
            raise ValueError("Failed to create a valid binary mask from template.")

        return template_mask

    def _calculate_iou(mask1, mask2):

        """Calculates Intersection over Union (IoU) between two binary masks."""
        intersection = cv2.bitwise_and(mask1, mask2)
        union = cv2.bitwise_or(mask1, mask2)
        iou = np.sum(intersection) / (np.sum(union) + 1e-6)
        return iou

    def align_naan_by_template_matching_fast(image, area, binary_mask, boundary,
                                                template_path_big=TEMPLATE_BIG,
                                                template_path_mini=TEMPLATE_MINI,
                                                scale_factor=0.13,  #size reduction to reduce pixels
                                                coarse_step=45,   # angle range it will check on 45, 90,... if coarse_step=45
                                                fine_range=0):    # +- in coarse step angle to check in case get better allign

        try:
            template_mask_full = _load_and_prep_template(template_path_big if area > 2000000 else template_path_mini)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error preparing template: {e}")
            return image, binary_mask, boundary

        # --- Downscale Masks ---
        template_small = cv2.resize(template_mask_full, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        input_small = cv2.resize(binary_mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

        # --- Find Centroids (on small masks) ---
        M_template = cv2.moments(template_small)
        if M_template['m00'] == 0:
            print("Warning: Template mask empty after scaling.")
            return image, binary_mask, boundary
        ct_x_s = int(M_template['m10'] / M_template['m00'])
        ct_y_s = int(M_template['m01'] / M_template['m00'])

        M_input = cv2.moments(input_small)
        if M_input['m00'] == 0:
            print("Warning: Input mask empty after scaling.")
            return image, binary_mask, boundary
        ci_x_s = int(M_input['m10'] / M_input['m00'])
        ci_y_s = int(M_input['m01'] / M_input['m00'])

        # --- Create Small Canvases ---
        h_t_s, w_t_s = template_small.shape
        h_i_s, w_i_s = input_small.shape
        diag_s = np.sqrt(max(h_t_s, h_i_s)**2 + max(w_t_s, w_i_s)**2)
        canvas_size_s = int(diag_s * 1.5)
        canvas_center_x_s, canvas_center_y_s = canvas_size_s // 2, canvas_size_s // 2
        canvas_center_s = (canvas_center_x_s, canvas_center_y_s)

        template_canvas_s = np.zeros((canvas_size_s, canvas_size_s), dtype=np.uint8)
        template_canvas_s[canvas_center_y_s - ct_y_s : canvas_center_y_s - ct_y_s + h_t_s,
                          canvas_center_x_s - ct_x_s : canvas_center_x_s - ct_x_s + w_t_s] = template_small

        input_canvas_s = np.zeros((canvas_size_s, canvas_size_s), dtype=np.uint8)
        input_canvas_s[canvas_center_y_s - ci_y_s : canvas_center_y_s - ci_y_s + h_i_s,
                      canvas_center_x_s - ci_x_s : canvas_center_x_s - ci_x_s + w_i_s] = input_small

        # --- Coarse Search ---
        best_iou = -1.0
        best_angle = 0.0
        for angle_int in range(0, 360, coarse_step):
            angle = float(angle_int)
            M_rot = cv2.getRotationMatrix2D(canvas_center_s, angle, 1.0)
            rotated_input = cv2.warpAffine(input_canvas_s, M_rot, (canvas_size_s, canvas_size_s), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            iou = _calculate_iou(template_canvas_s, rotated_input)
            if iou > best_iou:
                best_iou = iou
                best_angle = angle

        # --- Fine Search ---
        for angle_offset in range(-fine_range, fine_range + 1):
            angle = (best_angle + angle_offset) % 360.0 # Handle wrap around
            M_rot = cv2.getRotationMatrix2D(canvas_center_s, angle, 1.0)
            rotated_input = cv2.warpAffine(input_canvas_s, M_rot, (canvas_size_s, canvas_size_s), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            iou = _calculate_iou(template_canvas_s, rotated_input)
            if iou > best_iou:
                best_iou = iou
                best_angle = angle

        # print(f"Best angle found: {best_angle:.2f} degrees with IoU: {best_iou:.4f}")

        # --- Final Rotation (Full Res) ---
        M_input_orig = cv2.moments(binary_mask)
        if M_input_orig['m00'] == 0:
            print("Warning: Original input mask empty.")
            return image, binary_mask, boundary
        ci_x_orig = int(M_input_orig['m10'] / M_input_orig['m00'])
        ci_y_orig = int(M_input_orig['m01'] / M_input_orig['m00'])
        h_orig, w_orig = image.shape[:2]
        M_final = cv2.getRotationMatrix2D((ci_x_orig, ci_y_orig), best_angle, 1.0)

        rotated_image = cv2.warpAffine(image, M_final, (w_orig, h_orig), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rotated_mask = cv2.warpAffine(binary_mask, M_final, (w_orig, h_orig), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rotated_boundary = cv2.warpAffine(boundary, M_final, (w_orig, h_orig), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return rotated_image, rotated_mask, rotated_boundary


    def get_largest_contour(boundary_img):

        contours, _ = cv2.findContours(boundary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No contours found in image.")
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        return largest, area


    def find_tip(binary_image):

        rows = np.any(binary_image > 0, axis=1)
        rows_with_pixel = np.where(rows)[0]

        if len(rows_with_pixel) == 0:
            raise ValueError("No visible pixels found in the image.")

        tip_y = rows_with_pixel[0]
        tip_row = binary_image[tip_y]

        tip_x_coords = np.where(tip_row > 0)[0]
        tip_x = int(np.mean(tip_x_coords))

        return np.array([tip_x, tip_y])


    def find_base_points(contour, horiz_thresh=10, bottom_frac=0.3):

        diffs = np.roll(contour, -1, axis=0) - contour
        angles = np.degrees(np.arctan2(diffs[:, 1], diffs[:, 0]))
        is_horiz = (np.abs(angles) < horiz_thresh) | (np.abs(np.abs(angles) - 180) < horiz_thresh)
        ys = contour[:, 1]
        min_y, max_y = np.min(ys), np.max(ys)
        is_bottom = ys >= min_y + (1 - bottom_frac) * (max_y - min_y)
        base_mask = is_horiz & is_bottom

        return contour[base_mask]


    def fit_line(points):

        x = points[:, 0]
        y = points[:, 1]
        A = np.vstack([x, np.ones_like(x)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]

        return m, b

    def perpendicular_line(m, tip):

        if m == 0:
            return None, tip[0]

        elif np.isinf(m):
            return 0, tip[1]

        m_perp = -1 / m
        b_perp = tip[1] - m_perp * tip[0]

        return m_perp, b_perp

    def rotate_image(image, angle):

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

        return rotated, M

    def rotate_points(points, M):

        points = np.array(points, dtype=np.float32)
        if points.ndim == 1:
            points = np.expand_dims(points, axis=0)
        points = np.hstack([points, np.ones((points.shape[0], 1))])
        rotated = (M @ points.T).T

        return rotated

    def sample_extrema_from_binary_image(boundary, tip, num_lines=6, tolerance=10):

        max_y = np.max(boundary[:, 1])
        y_start = tip[1] + int((max_y - tip[1]) * 0.1)
        y_end = max_y - int((max_y - tip[1]) * 0.1)
        y_levels = np.linspace(y_start, y_end, num=num_lines + 1, dtype=int)
        boundary_x = boundary[:, 0]
        boundary_y = boundary[:, 1]
        left_points = []
        right_points = []
        for y in y_levels:
            mask = np.abs(boundary_y - y) <= tolerance
            candidates = boundary[mask]
            if candidates.shape[0] == 0:
                continue
            left_mask = candidates[:, 0] < tip[0]
            right_mask = ~left_mask
            if np.any(left_mask):
                left_candidates = candidates[left_mask]
                left_points.append(tuple(left_candidates[np.argmin(left_candidates[:, 0])]))
            if np.any(right_mask):
                right_candidates = candidates[right_mask]
                right_points.append(tuple(right_candidates[np.argmax(right_candidates[:, 0])]))

        return left_points, right_points, y_levels

    def compute_angles(points, side):

        angles = []
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            dx, dy = x2 - x1, y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))
            if side == "right":
                angle = angle % 360
            elif side == "left":
                angle = (angle + 180) % 360
            if angle > 180:
                angle = 360 - angle
            angles.append(round(angle, 2))

        return angles

    def show_base_curve_with_stats(contour,
                                  *,
                                  bottom_frac: float = 0.10,
                                  deg: int = 2,
                                  depth_thresh: int = 20,
                                  cluster_eps: int = 1,
                                  canvas_margin: int = 20) -> None:
        """
        Reports:
          • n_dents – count of bottom-edge points whose deviation > depth_thresh
        """
        cnt = np.asarray(contour).reshape(-1, 2).astype(np.int32)
        y_min, y_max = cnt[:, 1].min(), cnt[:, 1].max()
        band_y = y_max - bottom_frac * (y_max - y_min)
        base_pts = cnt[cnt[:, 1] >= band_y]
        if len(base_pts) <= deg:
            raise ValueError("Not enough base points to fit polynomial.")
        base_pts = base_pts[np.argsort(base_pts[:, 0])]
        xs, ys = base_pts[:, 0], base_pts[:, 1]
        coeffs = np.polyfit(xs, ys, deg)
        y_fit = np.polyval(coeffs, xs)
        residual = ys - y_fit
        over = residual > depth_thresh
        under = residual < -depth_thresh
        dent_mask = over | under
        dent_pts = base_pts[dent_mask]
        clusters = []
        remaining = dent_pts.copy()
        while len(remaining):
            seed = remaining[0]
            dists = np.linalg.norm(remaining - seed, axis=1)
            grp = remaining[dists < cluster_eps]
            clusters.append(tuple(np.mean(grp, axis=0)))
            remaining = remaining[dists >= cluster_eps]
        n_dents = len(clusters)

        return n_dents, xs, y_fit, clusters, coeffs

    def process_image(image):

        image = remove_bg(image)
        image, binary_mask, boundary = extract_boundary(image)
        _, area = get_largest_contour(boundary)

        total_rotation_time = time.time()
        image, binary_mask, boundary = align_naan_by_template_matching_fast(image, area, binary_mask, boundary)
        contour, _ = get_largest_contour(boundary)

        if contour is None:
            return None, None, None, None, None, None, None, "NOT OK"

        contour = contour.squeeze().astype(np.float32)

        base_points = find_base_points(contour)
        if len(base_points) < 2:
            return None, None, None, None, None, None, None, "NOT OK"

        m_base, b_base = fit_line(base_points)
        angle_rad = math.atan(m_base)
        angle_deg = np.degrees(angle_rad)
        stacked = np.stack([image, binary_mask, boundary], axis=2)  # shape: H×W×3

        # Compute rotation matrix M once:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        # One single call to warpAffine on the 3-channel stack:
        rotated_all = cv2.warpAffine(stacked, M, (w, h), flags=cv2.INTER_LINEAR)
        # After calling this, rotated_all has shape H×W×3.

        # Split back into three single‐channel images:
        rotated_image = rotated_all[:, :, 0]
        rotated_mask  = rotated_all[:, :, 1]
        rotated_boundary = rotated_all[:, :, 2]

        rotated_contour = rotate_points(contour, M)

        # Find tip in rotated mask and perpendicular axis
        tip = find_tip(rotated_mask)
        m_perp, b_perp = perpendicular_line(0, tip)

        # Sample left/right edge points and compute angles
        boundary_points = np.argwhere(rotated_boundary > 0)
        boundary_points = np.array([[pt[1], pt[0]] for pt in boundary_points], dtype=np.float32)
        left_pts, right_pts, _ = sample_extrema_from_binary_image(boundary_points, tip)
        left_angles = compute_angles(left_pts, "left")
        right_angles = compute_angles(right_pts, "right")

        # Pad left/right angles to exactly 6 values (if fewer points found, pad with 0)
        if len(left_angles) < 6:
            left_angles += [0] * (6 - len(left_angles))
        if len(right_angles) < 6:
            right_angles += [0] * (6 - len(right_angles))
        L = left_angles[:6]
        R = right_angles[:6]
        angles = L + R  # Combined list of 12 angles

        # 1. Left-Right angle differences
        angle_diffs = [abs(L[i] - R[i]) for i in range(6)]
        max_angle_diff = max(angle_diffs)

        # 2. Kinks: absolute change between consecutive angles
        left_kinks = [abs(L[i + 1] - L[i]) for i in range(4)]
        right_kinks = [abs(R[i + 1] - R[i]) for i in range(4)]
        all_kinks = left_kinks + right_kinks
        max_kink = max(all_kinks) if all_kinks else 0

        # 3. Folds: signed change between consecutive angles
        left_folds = [L[i + 1] - L[i] for i in range(4)]
        right_folds = [R[i + 1] - R[i] for i in range(4)]
        all_folds = left_folds + right_folds
        min_fold = min(all_folds) if all_folds else 0

        # Determine intermediate status after angle-based checks
        status_angles = "OK" if (max_angle_diff < MAX_ANGLE and max_kink < MAX_KINK and min_fold > MAX_FOLD) else "NOT OK"

        # Initialize values for base analysis
        n_dents = None
        xs = y_fit = clusters = coeffs = None

        # If angle checks passed, perform base dent analysis
        n_dents, xs, y_fit, clusters, coeffs = show_base_curve_with_stats(rotated_contour)

        # Determine final status after checking dents
        if status_angles == "OK" and (n_dents is not None and n_dents < MAX_DENTS):
            final_status = "OK"
        else:
            final_status = "NOT OK"

        is_DShaped = True if final_status == "OK" else False

        # Annotate image for PDF
        annotated = cv2.cvtColor(rotated_image, cv2.COLOR_GRAY2BGR)
        # Draw base clusters if available
        if clusters is not None:
            for pt in clusters:
                pt_int = tuple(map(int, pt))
                cv2.circle(annotated, pt_int, 10, (0, 0, 255), -1)

            # Draw polynomial with coeff
            x_curve = np.linspace(0, rotated_image.shape[1] - 1, 400)
            y_curve = np.polyval(coeffs, x_curve)
            curve_pts = np.column_stack((x_curve, y_curve)).astype(np.int32)
            cv2.polylines(annotated, [curve_pts.reshape(-1,1,2)], False, (0, 165, 255), 6)

        # Draw angles on left/right points
        angle_i = 0
        pts = left_pts + right_pts

        for i, pt in enumerate(pts):
            color = (255, 0, 0) if i < len(left_pts) else (0, 255, 0)
            pt_int = tuple(map(int, pt))
            cv2.circle(annotated, pt_int, 12, color, -1)

            if i > 0 and i != len(left_pts):
                prev_pt = tuple(map(int, pts[i - 1]))
                cv2.line(annotated, prev_pt, pt_int, color, 5)

            if i == 0 or i == len(left_pts):
                continue

            # Angle value with outline
            cv2.putText(annotated, f"{angles[angle_i]:.1f}", pt_int,
                        cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 0, 0), 10)
            cv2.putText(annotated, f"{angles[angle_i]:.1f}", pt_int,
                        cv2.FONT_HERSHEY_SIMPLEX, 5.0, color, 5)
            angle_i += 1

        # Draw base line and perpendicular axis
        x_vals = np.array([0, image.shape[1]])
        y_base = 0 * x_vals + b_base
        cv2.line(annotated, (int(x_vals[0]), int(y_base[0])), (int(x_vals[1]), int(y_base[1])), (0, 255, 255), 5)

        if m_perp is not None:
            y_perp = m_perp * x_vals + b_perp
            cv2.line(annotated, (int(x_vals[0]), int(y_perp[0])), (int(x_vals[1]), int(y_perp[1])), (255, 0, 255), 5)
        else:
            cv2.line(annotated, (int(b_perp), 0), (int(b_perp), image.shape[0]), (255, 0, 255), 5)

        # Overlay text with each step’s metrics
        text_x, text_y = 10, 30
        line_height = 30
        cv2.putText(annotated, f"MaxAngleDiff: {max_angle_diff:.1f}", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, f"MaxKink: {max_kink:.1f}", (text_x, text_y + line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, f"MinFold: {min_fold:.1f}", (text_x, text_y + 2*line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        dent_text = f"Dents: {n_dents}" if n_dents is not None else "Dents: N/A"
        cv2.putText(annotated, dent_text, (text_x, text_y + 3*line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, f"StatusAngles: {status_angles}", (text_x, text_y + 4*line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, f"FinalStatus: {final_status}", (text_x, text_y + 5*line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if final_status == "OK" else (0, 0, 255), 2)

        return {
            "Result": is_DShaped,
            "annotated_image": annotated
        }

    return process_image(image_input)

# Launch both threads
reader = threading.Thread(target=reader_thread, name="Reader", daemon=True)
worker = threading.Thread(target=worker_allocator, name="Worker")

reader.start()
worker.start()

# Wait for both to finish
reader.join()
worker.join()