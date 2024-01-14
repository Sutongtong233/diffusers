from diffusers import StableDiffusionPipeline
import torch
import os
import numpy as np
from PIL import Image

# token（训练的物体） train_seed（训练的seed） num_token  epoch（每500epochs保存一次） 共同决定了load哪个embedding
token = "cat_statue" # cat_statue elephant round_bird
train_seed = 42 # train时候的seed，用于选checkpoint
num_token = 2 # multi token的数量，比较合适
epoch = 3000 # 决定训练的过拟合程度（不过训练集5张图片，质量足够的话，基本不会太过拟合）500-3000
batch_size = 4 # 训练的batch size（取4的时候3090单卡极限了）
object = "Sketch"  # 希望讲定制化物体用于生成的物体，在下面的prompt_dict中选择

num_seeds = 16 # 生成16个随机seed用于inference
seeds = np.random.randint(0, 100000, size=num_seeds)
print("Generated seeds:", seeds)

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


# 用户输入和其他设置
prompt_dict = {
    "anime": f"Anime painting of a Kawaii <{token}>",
    "icon": f"App icon of <{token}>",
    "poster": f"A poster for <{token}>",
    "none": f"<{token}>",
    "painting": f"Painting of two <{token}> fishing on a boat",
    "Banksy": f"Banksy art of <{token}>",
    "Advertisement": f"Advertisement brochure of <{token}>",
    "Sketch": f"Sketch of <{token}>",
}
prompt = prompt_dict[object]

yourpath = "/home/stt/python_project/stt_data/stabilityai/"
model_id = yourpath + "stable-diffusion-v1-5"

# 加载模型
exp_path = f"/home/stt/py_github_repo_read/diffusers/examples/_outputs/TI_multi/{token}_seed_{train_seed}_numtoken_{num_token}"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.load_textual_inversion(exp_path, weight_name=f"learned_embeds-steps-{epoch}.bin", use_safetensors=False)

# 生成图片并保存为4x4网格
images = []
for seed in seeds:
    seed = int(seed)
    latents = getLatents(seed=seed)
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, latents=latents).images[0]
    images.append(image)

# 创建4x4网格
grid_width = 4
width = height = 512
grid = Image.new('RGB', (width * grid_width, height * grid_width))
for i, image in enumerate(images):
    grid.paste(image, (width * (i % grid_width), height * (i // grid_width)))

# 保存网格图片
os.makedirs(f"{exp_path}/_outputs/", exist_ok=True)
grid.save(f"{exp_path}/_outputs/{token}-{object}-{epoch}.png")


