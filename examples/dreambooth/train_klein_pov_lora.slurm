#!/bin/bash -l
#SBATCH --account=ecs
#SBATCH --partition=a100,swarm_a100,swarm_h100,l40,dual_h200,quad_h200
#SBATCH --job-name=klein_pov_lora
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1

# Submit from this directory (diffusers/examples/dreambooth):  sbatch train_klein_pov_lora.slurm
#
# One-time setup on the login node (compute nodes run offline):
#   export HF_HOME=/scratch/hgm1g14/cache_dir   # same cache as below, so the job finds the downloads
#   conda activate gemma && pip install -e ../.. && pip install -r requirements_flux3_klein.txt matplotlib
#   hf auth login        # the dataset is private
#   hf download black-forest-labs/FLUX.2-klein-base-9B
#   hf download harveymannering/camera-control-pairs --repo-type dataset

echo Script starting
nvidia-smi

export HF_HOME=/scratch/hgm1g14/cache_dir
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

conda activate gemma

OUT=/scratch/hgm1g14/FluxLoRAOutputs/flux2-klein-pov  # LoRA, checkpoints and validation grids
mkdir -p "$OUT"

# load_dataset can't resolve a Hub name offline, so point it at the downloaded snapshot directly
DATASET=$(ls -d "$HF_HOME"/hub/datasets--harveymannering--camera-control-pairs/snapshots/*/ | head -n 1)

# Every 500 steps the 20 emotic_test_20/*_annotated.png images are run through the model and saved as
# $OUT/validation_step_<step>.png (input | POV grid); the final LoRA's grid is $OUT/test_step_<step>.png.
accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no \
  train_dreambooth_lora_flux2_klein_img2img.py \
  --pretrained_model_name_or_path=/iridisfs/scratch/hgm1g14/cache_dir/hub/models--black-forest-labs--FLUX.2-klein-base-9B/snapshots/32773329fbe7e81a90ef971740e8ba4b0364ecf3 \
  --output_dir="$OUT" \
  --dataset_name="$DATASET" \
  --image_column="image" --cond_image_column="cond_image" --caption_column="caption" \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --cache_latents --max_sequence_length=64 \
  --resolution=512 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="adamw" \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=200 \
  --max_train_steps=10000 \
  --checkpointing_steps=500 \
  --rank=128 \
  --validation_prompt="Make the camera the POV perspective of the person in the highlighted person, high quality photo" \
  --validation_image=emotic_test_20 \
  --validation_steps=500 \
  --seed="0"

nvidia-smi
echo Script completed =3
