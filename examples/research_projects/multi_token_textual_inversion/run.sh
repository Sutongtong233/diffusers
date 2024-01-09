export OBJECT="round_bird" # round_bird, elephant, cat_statue
export SEED=42
export PRIOR_CLASS="statue" # statue
export MODEL_NAME="/home/stt/python_project/stt_data/runwayml/stable-diffusion-v1-5"
export DATA_DIR="/home/stt/py_github_repo_read/diffusers/examples/_inputs/TI_history_data/${OBJECT}"
export NUM_TOKEN=2
export OUTPUT_DIR="/home/stt/py_github_repo_read/diffusers/examples/_outputs/TI_multi/${OBJECT}_seed_${SEED}_numtoken_${NUM_TOKEN}/"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<$OBJECT>" --initializer_token=$PRIOR_CLASS \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --num_vec_per_token $NUM_TOKEN