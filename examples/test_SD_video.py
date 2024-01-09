from diffusers import StableVideoDiffusionPipeline # image-to-video

import torch
from diffusers import DiffusionPipeline, TextToVideoSDPipeline
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")

prompt = "Spiderman is surfing"
video_frames = pipe(prompt).frames
video_path = export_to_video(video_frames, output_video_path="/home/stt/py_github_repo_read/diffusers/examples/_outputs/text_to_video/Spiderman.mp4")




pipe = TextToVideoSDPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
prompt = "Spiderman is surfing"
video_frames = pipe(prompt).frames
video_path = export_to_video(video_frames, output_video_path="/home/stt/py_github_repo_read/diffusers/examples/_outputs/text_to_video/Spiderman.mp4")