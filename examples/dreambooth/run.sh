export MODEL_NAME="/home/stt/python_project/stt_data/runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/stt/py_github_repo_read/diffusers/examples/_inputs/dunhuang"
export OUTPUT_DIR="/home/stt/py_github_repo_read/diffusers/examples/_outputs/DB/dunhuang/"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A app icon of sks" \
  --validation_epochs=50 \
  --seed="0" \
#   --push_to_hub
#   --report_to="wandb" \