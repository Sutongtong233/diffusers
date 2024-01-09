from diffusers import StableDiffusionPipeline
import torch
import os

# base model可以任选（SD1.5/SD2.1/SDXL），但是inference要保证和train的时候一致
# 这里选择SD1.5 （非sota 比较适合初期试验阶段）  
# TODO 需要提前下载, 然后输入自己的路径 
yourpath = "/home/stt/python_project/stt_data/stabilityai/"
model_id = yourpath + "stable-diffusion-v1-5"  
train_seed = 0
personal_object = "cat_statue"  # round_bird, elephant, cat_statue
model_id = f"/home/stt/py_github_repo_read/diffusers/examples/_outputs/DB/{personal_object}_seed_{train_seed}/"
seed = 1314  # inference seed
object = "mug" # inference 想要生成的东西

# 用户输入 需要生成的物体。sks {prior_object}是训练的时候就指定好的，每个定制化物体一个单独的<xxx>的特殊token
prior_object = "toy"
prompt_dict = {
    "anime": f"Anime painting of a Kawaii sks {prior_object}",
    "icon": f"App icon of sks {prior_object}",
    "poster": f"A poster for sks {prior_object}",
    "none": f"sks {prior_object}",
    "painting": f"Painting of two sks {prior_object} fishing on a boat",
    "Banksy": f"Banksy art of sks {prior_object}",
    "T-shirt_1": f"T-shirt of sks {prior_object}",
    "T-shirt_2": f"sks {prior_object} on a T-shirt",
    "mug": f"sks {prior_object} on a mug",
}


def getLatents(num_images=1, height=512, width=512, seed=42, device="cuda"):  # SD1.5只能生成512 512图片，
    generator = torch.Generator(device=device)
    generator = generator.manual_seed(seed)
    latents = torch.randn(
        (num_images, 4, height // 8, width // 8),
        generator = generator,
        device = device,
        dtype = torch.float16
    )
    return latents
latents = getLatents(seed=seed)  # diffusion初始化的随机噪声Z_T


prompt = prompt_dict[object]
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, latents=latents).to("cuda")

os.makedirs(f"{model_id}/_output_imgs/", exist_ok=True)
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save(f"{model_id}/_output_imgs/{object}-{seed}.png")