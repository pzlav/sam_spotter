import os
import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from decord import VideoReader, cpu
import decord
decord.bridge.set_bridge('torch')
from einops import rearrange

MOTION_IMAGE_THRESHOLD = 0.5
output_size = (640, 480)
video_dir = "/home/pzla/projects/final_DLS/recorded_clips"
output_dir = "/home/pzla/projects/recorded_clips_output"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")


if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


print(f"Loading SAM2 model")
from sam2.build_sam import build_sam2_video_predictor
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


# read the list of video files in the directory
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
#video_files = ["motion_clip_1736768912.mp4"]

for i in range(len(video_files)):
    video_file = video_files[i]
    print(f"Processing video: {video_file}")
    video_reader = VideoReader(os.path.join(video_dir, video_file), ctx=cpu(0))
    fps = video_reader.get_avg_fps()
    video_np = video_reader.get_batch(list(range(len(video_reader))))
    #video = torch.tensor(video_np, dtype=torch.float32) / 255.0
    video = video_np.float() / 255.0
    video = rearrange(video,'t h w c -> t c h w')
    video = rearrange(video, 't c h w -> c t h w').unsqueeze(0).half()
    video = video.cuda()
    print(f"Video shape: {video.shape}")

    frame_1 = video[0, :, 0, :, :].cpu()  
    frame_3 = video[0, :, 2, :, :].cpu()  
    motion_image = torch.abs(frame_3 - frame_1)
    motion_image_np = motion_image.permute(1, 2, 0).numpy()
    motion_image_gray = motion_image.sum(dim=0)
    motion_mask = (motion_image_gray > MOTION_IMAGE_THRESHOLD).float() 
    non_zero_coords = torch.nonzero(motion_mask, as_tuple=False)
    if non_zero_coords.numel() > 0:  # Ensure there are non-zero pixels
        centroid = non_zero_coords.float().mean(dim=0)  # Shape: (2,)
        centroid_x, centroid_y = centroid[1].item(), centroid[0].item()  # (x, y)
    centroid_x, centroid_y = int(centroid_x), int(centroid_y)
    print(f"Centroid: ({centroid_x}, {centroid_y})")
    # Move centroid to the closest non-zero pixel if it is not inside the mask
    try:
        if motion_mask[centroid_y, centroid_x] == 0:
            distances = torch.cdist(centroid.unsqueeze(0), non_zero_coords.float())
            closest_idx = torch.argmin(distances)
            centroid_y, centroid_x = non_zero_coords[closest_idx].tolist()
            centroid_x, centroid_y = int(centroid_x), int(centroid_y)
            print(f"NEW Centroid: ({centroid_x}, {centroid_y})")
    except Exception as e:
        print("Error in new centroid calculation")
        print(e)
        continue


    
    with torch.no_grad():
        inference_state = predictor.init_state(os.path.join(video_dir, video_file))  
        predictor.reset_state(inference_state)
        ann_frame_idx = 0 
        ann_obj_id = 1 
        points = np.array([[centroid_x, centroid_y]], dtype=np.float32)
        labels = np.array([1], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }


    out1_list = []
    out2_list = []

    for frame_idx in range(len(video_segments)-1):
        frame = video[0, :, frame_idx, :, :]  # [3, 480, 640]
        frame_nump = (frame * 255.0).cpu().numpy() 
        frame_tr = np.transpose(frame_nump, (1, 2, 0))  # Convert to [H, W, C] for OpenCV (RGB format)
        frame_tr = frame_tr.astype(np.uint8)  
        frame_tr = cv2.cvtColor(frame_tr, cv2.COLOR_RGB2BGR)
        mask_frame_dict = video_segments.get(frame_idx, {})
        try:
            mask_dict = mask_frame_dict[1][0,:,:] #(480, 640)
        except:
            mask_dict = np.zeros((480, 640))
        mask_rgb = np.stack((mask_dict, mask_dict, mask_dict), axis=-1).astype(np.uint8) * 255
        blended_frame = cv2.addWeighted(frame_tr, 0.7, mask_rgb, 0.3, 0)  # Blend with transparency
        out2_list.append(blended_frame)
        frame_tr = frame_tr * mask_dict[..., None]
        frame_tr[mask_dict != 0] = 255
        out1_list.append(frame_tr)
    


    video_file_name_wo_ext = os.path.splitext(video_file)[0]
    output_filename1 = os.path.join(output_dir, f"{video_file_name_wo_ext}_sam.mp4")
    out1 = cv2.VideoWriter(output_filename1, cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)
    for frame in out1_list:
        out1.write(frame)
    out1.release()
    output_filename2 = os.path.join(output_dir, f"{video_file_name_wo_ext}_blend.mp4")
    out2 = cv2.VideoWriter(output_filename2, cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)
    for frame in out2_list:
        out2.write(frame)
    out2.release()
    print(f"Video saved as {output_filename1}\n")