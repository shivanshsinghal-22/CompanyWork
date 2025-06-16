import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import cv2
from segmentation_models_pytorch import DeepLabV3Plus
import time
import math





GRID_ROWS = 2
GRID_COLS = 2








def merge_close_contours(contours, distance_thresh=100):
    merged = []
    used = [False] * len(contours)

    for i in range(len(contours)):
        if used[i]:
            continue
        cnt1 = contours[i]
        x1, y1, w1, h1 = cv2.boundingRect(cnt1)
        cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2

        merged_cnt = cnt1.copy()

        for j in range(i + 1, len(contours)):
            if used[j]:
                continue
            cnt2 = contours[j]
            x2, y2, w2, h2 = cv2.boundingRect(cnt2)
            cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2

            dist = math.hypot(cx1 - cx2, cy1 - cy2)
            if dist < distance_thresh:
                merged_cnt = np.vstack((merged_cnt, cnt2))
                used[j] = True

        used[i] = True
        merged.append(merged_cnt)

    return merged









class PatchImageDataset(Dataset):
    def __init__(self, image, transform=None):
        self.transform = transform
        self.original_image = image
        self.image_np = np.array(self.original_image)
        self.h, self.w = self.image_np.shape[:2]
        self.patches = self._create_patches()
    def _create_patches(self):
        patches = []
        step_h = self.h // GRID_ROWS
        step_w = self.w // GRID_COLS
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                patch = self.original_image.crop((j * step_w, i * step_h, (j+1) * step_w, (i+1) * step_h))
                patch = patch.resize((1024, 1024))
                patches.append(patch)
        return patches
    def __len__(self):
        return len(self.patches)
    def __getitem__(self, idx):
        patch = self.patches[idx]
        if self.transform:
            patch = self.transform(patch)
        return patch
transform = transforms.Compose([
    transforms.ToTensor()
])
# Load model for patch-level prediction (hair detection)
hair_model = DeepLabV3Plus(encoder_name="timm-efficientnet-b4", encoder_weights=None, in_channels=3, classes=1)
hair_model.load_state_dict(torch.load("checkpoints_dlv3/deeplabv3_efficientnetb4_epoch200.pth"))
hair_model = hair_model.to("cuda")
# hair_model_0 = hair_model.to("cuda:0")
# hair_model_1 = hair_model.to("cuda:1")
# hair_model_0.eval()
# hair_model_1.eval()
hair_model.eval()
# Load model for full-image boundary detection (naan detection)
naan_model = DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
naan_model.load_state_dict(torch.load("deeplabv3_seg_model.pth"))
naan_model = naan_model.to("cuda")
# naan_model_0 = naan_model.to("cuda:0")
# naan_model_1 = naan_model.to("cuda:1")
# naan_model_0.eval()
naan_model.eval()
os.makedirs("overlay_full", exist_ok=True)
os.makedirs("side_by_side_full", exist_ok=True)
os.makedirs("side_by_side_naan", exist_ok=True)






def show_result_window(process_res,save_prefix):
    process_res_np = process_res.copy()
    process_res_np = cv2.resize(process_res_np, (640, 480))
    cv2.imshow(f"Processed Result_{save_prefix}", process_res_np)
    cv2.waitKey(5000)  # Show for 5 seconds (5000 ms)
    cv2.destroyWindow(f"Processed Result_{save_prefix}")
# Camera setup





def process(frame_pil, save_prefix="result"):
    original_image = frame_pil.copy()
    dataset = PatchImageDataset(frame_pil, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    start = time.time()
    patches_pred = []
    with torch.no_grad():
        for patch in loader:
            patch = patch.to(f"cuda")
            output = hair_model(patch)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask_bin = (pred_mask > 0.4).astype(np.uint8) * 255
            patches_pred.append(pred_mask_bin)
        print(f"time per image is = {time.time()-start}")
    patch_size = 1024
    full_mask = np.zeros((patch_size * GRID_ROWS, patch_size * GRID_COLS), dtype=np.uint8)
    k = 0
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            full_mask[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patches_pred[k]
            k += 1
    original_np = cv2.cvtColor(np.array(original_image.resize((patch_size * GRID_COLS, patch_size * GRID_ROWS))), cv2.COLOR_RGB2BGR)
    # Use naan model to get boundary mask from full image
    image_tensor = transform(Image.fromarray(original_np[:, :, ::-1])).unsqueeze(0).to(f"cuda")
    with torch.no_grad():
        boundary_output = naan_model(image_tensor)
        boundary_mask = torch.sigmoid(boundary_output).squeeze().cpu().numpy()
        boundary_mask_bin = (boundary_mask > 0.8).astype(np.uint8) * 255
    # Save side-by-side of original + naan mask
    naan_mask_vis = cv2.cvtColor(boundary_mask_bin, cv2.COLOR_GRAY2BGR)
    side_by_side_naan = np.concatenate((original_np, naan_mask_vis), axis=1)
    cv2.imwrite(f"side_by_side_naan/{save_prefix}_naan_sidebyside.png", side_by_side_naan)
    # Extract thick boundary from naan mask
    contours, _ = cv2.findContours(boundary_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    naan_boundary_mask = np.zeros_like(boundary_mask_bin)
    cv2.drawContours(naan_boundary_mask, contours, -1, 255, thickness=20)  # Thicker boundary
    # Remove pixels in full_mask that coincide with naan boundary
    overlap = cv2.bitwise_and(full_mask, naan_boundary_mask)
    full_mask[overlap > 0] = 0
    # Remove small white regions (< 500 px)
    small_contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_mask = np.zeros_like(full_mask)
    for cnt in small_contours:
        if cv2.contourArea(cnt) >= 900:
            cv2.drawContours(cleaned_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    full_mask = cleaned_mask
    # Visualize: red boundary over white detections
    vis_image = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)
    vis_image[naan_boundary_mask > 0] = [0, 0, 255]  # Red boundary
    # Save outputs
    cv2.imwrite(f"overlay_full/{save_prefix}_overlay.png", vis_image)
    cv2.imwrite(f"overlay_full/{save_prefix}.png", vis_image)
    cv2.imwrite(f"side_by_side_full/{save_prefix}_sidebyside.png", np.concatenate((original_np, vis_image), axis=1))
    print(f":white_check_mark: Processed and saved {save_prefix}")
    # show_result_window(full_mask,save_prefix)
    # Detect bounding boxes around white areas (hair) in full_mask
    # Define corner positions and track used ones
    corner_positions = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
    used_corners = []

    image_h, image_w = original_np.shape[:2]
    margin = 20  # space from edge

    # Detect and merge close contours
    hair_contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged_contours = merge_close_contours(hair_contours, distance_thresh=100)

    for idx, cnt in enumerate(merged_contours):
        if idx >= 4:
            break  # only handle 4 corners

        x, y, w, h = cv2.boundingRect(cnt)
        hair_crop = original_np[y:y + h, x:x + w]

        # Resize to 3x
        new_w = int(w * 3)
        new_h = int(h * 3)
        resized_crop = cv2.resize(hair_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Assign corner
        corner = corner_positions[idx]
        used_corners.append(corner)

        if corner == 'top-left':
            paste_x = margin
            paste_y = margin
        elif corner == 'top-right':
            paste_x = image_w - new_w - margin
            paste_y = margin
        elif corner == 'bottom-left':
            paste_x = margin
            paste_y = image_h - new_h - margin
        elif corner == 'bottom-right':
            paste_x = image_w - new_w - margin
            paste_y = image_h - new_h - margin

        # Paste enlarged crop at corner
        original_np[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = resized_crop

        # Draw red bbox on original detection
        cv2.rectangle(original_np, (x, y), (x + w, y + h), (0, 255, 255), 8)

        # Draw red box around zoomed-in crop
        cv2.rectangle(original_np, (paste_x, paste_y), (paste_x + new_w, paste_y + new_h), (0, 0, 255), 10)

        # Draw dotted line from original bbox center to pasted image center (optional)
        orig_center = (x + w // 2, y + h // 2)
        paste_center = (paste_x + new_w // 2, paste_y + new_h // 2)

        num_dots = 30
        # for i in range(num_dots):
        #     dot_x = int(orig_center[0] + (paste_center[0] - orig_center[0]) * i / num_dots)
        #     dot_y = int(orig_center[1] + (paste_center[1] - orig_center[1]) * i / num_dots)
        #     cv2.circle(original_np, (dot_x, dot_y), 5, (0, 0, 0), -1)

    # Save final result



       # Define text and color based on detection
    if len(merged_contours) == 0:
        label_text = "Check OK"
        text_color = (0, 255, 0)  # Green
    else:
        label_text = "Hair Detected"
        text_color = (0, 0, 255)  # Red

    # Text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4.0
    thickness = 15
    margin = 30

    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

    # Bottom-left position
    x = margin
    y = original_np.shape[0] - margin

    # Background rectangle
    cv2.rectangle(
        original_np,
        (x - 10+500, y - text_height - 10),
        (x +500 + text_width + 10, y + 10),
        (255, 255, 255),  # White background
        -1  # Filled
    )

    # Draw text on top
    cv2.putText(
        original_np,
        label_text,
        (x+500, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )


    # Save final result
    cv2.imwrite(f"side_by_side_full/{save_prefix}_zoomed.png", original_np)
    # original_np = cv2.cvtColor(original_np, cv2.COLOR_BGR2RGB)




    cv2.imwrite(f"side_by_side_full/{save_prefix}_zoomed.png", original_np)

    original_np = cv2.cvtColor(original_np, cv2.COLOR_BGR2RGB)
    return original_np,full_mask
if __name__ == "__main__":
    test_folder = "dataset/train/images"
    os.makedirs(test_folder, exist_ok=True)
    for fname in os.listdir(test_folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(test_folder, fname)
            image = Image.open(img_path).convert("RGB")
            save_name = os.path.splitext(fname)[0]
            process(image, save_prefix=save_name)


















