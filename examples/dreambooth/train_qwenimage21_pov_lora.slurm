#!/bin/bash -l
#SBATCH --account=ecs
#SBATCH --partition=a100,swarm_a100,swarm_h100,l40,dual_h200,quad_h200
#SBATCH --job-name=qwen21_pov_lora
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1

# Submit from this directory (diffusers/examples/dreambooth):  sbatch train_qwenimage21_pov_lora.slurm
#
# One-time setup on the login node (compute nodes run offline):
#   export HF_HOME=/scratch/hgm1g14/cache_dir   # same cache as below, so the job finds the downloads
#   conda activate gemma && pip install -e ../.. && pip install -r requirements_flux.txt matplotlib bitsandbytes
#   hf auth login        # the dataset is private
#   hf download Qwen/Qwen-Image-2.1
#   hf download harveymannering/camera-control-pairs --repo-type dataset

echo Script starting
nvidia-smi

export HF_HOME=/scratch/hgm1g14/cache_dir
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

conda activate gemma

OUT=/scratch/hgm1g14/QwenLoRAOutputs/qwenimage21-pov  # LoRA, checkpoints and validation grids
mkdir -p "$OUT"

# Neither name resolves offline, so point at the downloaded snapshots directly
MODEL=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen-Image-2.1/snapshots/*/ | head -n 1)
DATASET=$(ls -d "$HF_HOME"/hub/datasets--harveymannering--camera-control-pairs/snapshots/*/ | head -n 1)

PROMPT="Make the camera the POV perspective of the person in the highlighted person, high quality photo"

# Before training and then every 500 steps, the 20 emotic_test_20/*_annotated.png images are run through the model
# and saved as $OUT/validation_step_<step>.png (input | POV grid), starting with validation_step_0.png for the
# untrained model; the final LoRA's grid is $OUT/test_step_<step>.png.
# Qwen-Image 2.1 notes (README_qwenimage21.md): --resolution must be a multiple of 32 and at least 256; there is no
# --max_sequence_length or --guidance_scale; keep --train_batch_size=1 since a batch must share one image-pad layout;
# --instance_prompt is required and only used for samples whose caption is empty.
accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no \
  train_dreambooth_lora_qwenimage21_img2img.py \
  --pretrained_model_name_or_path="$MODEL" \
  --output_dir="$OUT" \
  --dataset_name="$DATASET" \
  --image_column="image" --cond_image_column="cond_image" --caption_column="caption" \
  --instance_prompt="$PROMPT" \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --cache_latents \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="adamw" \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=200 \
  --max_train_steps=10000 \
  --checkpointing_steps=500 \
  --rank=128 \
  --lora_alpha=128 \
  --validation_prompt="$PROMPT" \
  --validation_image=emotic_test_20 \
  --validation_steps=500 \
  --seed="0"

nvidia-smi
echo Script completed =3
