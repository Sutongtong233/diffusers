export OBJECT="cat_statue" # round_bird, elephant, cat_statue
export SEED=42
export PRIOR_CLASS="toy"
export MODEL_NAME="/home/stt/python_project/stt_data/runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/stt/py_github_repo_read/diffusers/examples/_inputs/TI_history_data/${OBJECT}"
export OUTPUT_DIR="/home/stt/py_github_repo_read/diffusers/examples/_outputs/DB/${OBJECT}_seed_${SEED}/"
export CLASS_DIR="/home/stt/py_github_repo_read/diffusers/examples/_outputs/DB/${PRIOR_CLASS}_prior_imgs"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks ${PRIOR_CLASS}" \
  --class_prompt="a photo of ${PRIOR_CLASS}" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --seed ${SEED}

# export MODEL_NAME="/home/stt/python_project/stt_data/runwayml/stable-diffusion-v1-5"
# export INSTANCE_DIR="/home/stt/py_github_repo_read/diffusers/examples/_inputs/TI_history_data/elephant"
# export OUTPUT_DIR="/home/stt/py_github_repo_read/diffusers/examples/_outputs/DB_lora/elephant_seed_42/"
# export CLASS_DIR="/home/stt/py_github_repo_read/diffusers/examples/_outputs/DB_lora/statue_prior_imgs"

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a photo of sks toy" \
#   --class_prompt="a photo of toy" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-4 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="A photo of sks toy in a bucket" \
#   --validation_epochs=50 \
#   --seed="0" \

