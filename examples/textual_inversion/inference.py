from diffusers import StableDiffusionPipeline
import torch

# token（训练的物体） train_seed（训练的seed） 和 epoch（每500epochs保存一次） 共同决定了load哪个embedding
token = "cat_statue" # cat_statue elephant round_bird
train_seed = 42 # 我争取找到更稳定的seed
epoch = 3000 # 决定训练的过拟合程度（不过训练集5张图片，质量足够的话，基本不会太过拟合）
batch_size = 4 # 训练的batch size（取2的时候3090单卡极限了）
# inference的seed，可以参考webui的操作：用户选择/-1代表随机
seed = 23

# 用户输入 需要生成的物体。<{token}>是训练的时候就指定好的，每个定制化物体一个单独的<xxx>的特殊token
prompt_dict = {
    "anime": f"Anime painting of a Kawaii <{token}>",
    "icon": f"App icon of <{token}>",
    "poster": f"A poster for <{token}>",
    "none": f"<{token}>",
    "painting": f"Painting of two <{token}> fishing on a boat",
    "Banksy": f"Banksy art of <{token}>"
}

# 为了保存图片的命名
object = "Banksy"
prompt = prompt_dict[object]

# base model可以任选（SD1.5/SD2.1/SDXL），但是inference要保证和train的时候一致
# 这里选择SD1.5 （非sota 比较适合初期试验阶段）  
# TODO 需要提前下载, 然后输入自己的路径 
yourpath = "/home/stt/python_project/stt_data/stabilityai/"
model_id = yourpath + "stable-diffusion-v1-5"  

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

exp_path = f"/home/stt/py_github_repo_read/diffusers/examples/_outputs/TI_SD1.5/{token}_seed_{train_seed}_bs_{batch_size}/"
latents = getLatents(seed=seed)  # diffusion初始化的随机噪声Z_T
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.load_textual_inversion(exp_path, epoch=epoch)
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, latents=latents).images[0]
image.save(f"{exp_path}/{token}-{object}-{epoch}-{seed}.png")

