export MODEL_NAME="/home/stt/python_project/stt_data/stabilityai/stable-diffusion-2-1-base"
export DATA="9_colordeer"
export DATA_DIR="/home/stt/py_github_repo_read/diffusers/examples/_inputs/TI_history_data/$DATA"
export OBJECT="app icon"
export PROMPT="an $OBJECT of <$DATA>"
export STEP=10
export MAX_STEP=3000
export SEED=42
export BATCH=4

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --initializer_token="Chinese" \
  --placeholder_token="<$DATA>" \
  --resolution=512 \
  --train_batch_size=$BATCH \
  --gradient_accumulation_steps=4 \
  --max_train_steps=$MAX_STEP \
  --validation_steps=$STEP \
  --checkpointing_steps=$STEP \
  --save_steps=$STEP \
  --validation_prompt="" \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="/home/stt/py_github_repo_read/diffusers/examples/_outputs/TI_SD2.1/${DATA}_seed_${SEED}" \
  --seed $SEED

