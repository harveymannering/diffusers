#!/bin/bash -l
#SBATCH --account=ecs
#SBATCH --partition=a100,swarm_a100,swarm_h100,l40,dual_h200,quad_h200
#SBATCH --job-name=qwen21_emotic_pov
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-15

# Submit from this directory (diffusers/examples/dreambooth):  sbatch generate_emotic_pov_qwenimage21.slurm
# Each array task takes every 16th EMOTIC image (34k people in total). Resubmitting skips finished people.
# Quick test (one task, 10 images):  sbatch --array=0 generate_emotic_pov_qwenimage21.slurm --max_images 10
# Any arguments after the script name are passed on, e.g. --num_images_per_person 4 or --annotation box.
#
# One-time setup on the login node, on top of train_qwenimage21_pov_lora.slurm's (compute nodes run offline):
#   export TORCH_HOME=/scratch/hgm1g14/cache_dir/torch
#   conda activate gemma && pip install scipy
#   python -c "import torchvision; torchvision.models.detection.keypointrcnn_resnet50_fpn(weights='COCO_V1')"

echo Script starting
nvidia-smi

export HF_HOME=/scratch/hgm1g14/cache_dir
export TORCH_HOME=/scratch/hgm1g14/cache_dir/torch
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

conda activate gemma

MODEL=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen-Image-2.1/snapshots/*/ | head -n 1)
LORA=/scratch/hgm1g14/QwenLoRAOutputs/qwenimage21-pov/pytorch_lora_weights.safetensors
OUT=/scratch/hgm1g14/QwenLoRAOutputs/emotic-pov

python generate_emotic_pov_qwenimage21.py \
  --emotic_root=/scratch/hgm1g14/EMOTIC_PAMI \
  --pretrained_model_name_or_path="$MODEL" \
  --lora_path="$LORA" \
  --output_dir="$OUT" \
  --prompt="Make the camera the POV perspective of the person in the highlighted person, high quality photo" \
  --num_inference_steps=40 \
  --true_cfg_scale=1.0 \
  --num_images_per_person=1 \
  --shard="${SLURM_ARRAY_TASK_ID:-0}" \
  --num_shards="${SLURM_ARRAY_TASK_COUNT:-1}" \
  "$@"

nvidia-smi
echo Script completed =3
