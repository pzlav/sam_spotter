from models.modeling_vae import CVVAEModel
from decord import VideoReader, cpu
import torch
import os
from einops import rearrange
from torchvision.io import write_video
from torchvision import transforms 
from fire import Fire

def main(vae_path, video_path1, video_path2, save_path, height=576//2, width=1024//2):
    vae3d = CVVAEModel.from_pretrained(vae_path,subfolder="vae3d",torch_dtype=torch.float16)
    vae3d.requires_grad_(False)

    transform = transforms.Compose([
        transforms.Resize(size=(height,width))
    ])

    vae3d = vae3d.cuda()
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    video_reader = VideoReader(video_path1,ctx=cpu(0))

    fps = video_reader.get_avg_fps()

    video = video_reader.get_batch(list(range(len(video_reader)))).asnumpy()

    video = rearrange(torch.tensor(video),'t h w c -> t c h w')

    video = transform(video)

    video = rearrange(video,'t c h w -> c t h w').unsqueeze(0).half()

    frame_end = 1 + (len(video_reader) -1) // 4 * 4

    video = video / 127.5 - 1.0

    video= video[:,:,:frame_end,:,:]

    video = video.cuda()
    video[:, 1:, :, :, :] = 0.0
    print(f'Shape of input video: {video.shape}')
    latent = vae3d.encode(video).latent_dist.sample()

    print(f'Shape of video latent: {latent.shape}')
    b, c, t, h, w = latent.shape
    reverse_latent = latent.clone()
    reverse_latent = torch.flip(reverse_latent, dims=[2]) 
    # latent_diff = reverse_latent - latent

    # video_reader2 = VideoReader(video_path2,ctx=cpu(0))
    # video2 = video_reader2.get_batch(list(range(len(video_reader2)))).asnumpy()
    # video2 = torch.tensor(video2)
    # video2 = rearrange(video2,'t h w c -> t c h w')
    # video2 = transform(video2)
    # video2 = rearrange(video2,'t c h w -> c t h w').unsqueeze(0).half()
    # frame_end = 1 + (len(video_reader2) -1) // 4 * 4
    # video2 = video2 / 127.5 - 1.0
    # video2 = video2[:,:,:frame_end,:,:]
    # video2 = video2.cuda()
    # latent2 = vae3d.encode(video2).latent_dist.sample()

    # new_latent = (latent2 + latent) / 2




    results = vae3d.decode(reverse_latent).sample

    results = rearrange(results.squeeze(0), 'c t h w -> t h w c')

    results = (torch.clamp(results,-1.0,1.0) + 1.0) * 127.5
    results = results.to('cpu', dtype=torch.uint8)

    print(f"Results dtype: {results.dtype}, shape: {results.shape}")

    write_video(save_path, results,fps=fps,options={'crf': '10'})

if __name__ == '__main__':
    main(
        vae_path="/home/pzla/.cache/huggingface/hub/models--AILab-CVC--CV-VAE/snapshots/cbfb510e66801521cacbc598d65f8a6eb3075884/",
        video_path1="/home/pzla/projects/recorded_clips_output/1_test/motion_clip_1737288085_sam.mp4",
        #video_path1="/home/pzla/projects/sam2/output_cut_out.mp4",
        video_path2="/home/pzla/projects/final_DLS/recorded_clips/motion_clip_1736083123.mp4",
        save_path="data/test.mp4",
    )

