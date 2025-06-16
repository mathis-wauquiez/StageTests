#!/bin/bash
#SBATCH --job-name=line-int-flow
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --partition=A100,A40,P100
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mathis.wauquiez@eleves.enpc.fr
#SBATCH --array=0-4

IMAGE_PATHS=(
    data/example_images/1.png
    data/example_images/2.png
    data/example_images/bricks.jpg
    data/example_images/grass.jpg
    data/example_images/horizontal_gradient.png
)
MASK_PATHS=(
    data/example_images/mask2.png
    data/example_images/mask3.png
    data/example_images/mask.png
    data/example_images/grass_mask.png
    data/example_images/mask.png
)

if [ "${#IMAGE_PATHS[@]}" -ne "${#MASK_PATHS[@]}" ]; then
  echo "IMAGE_PATHS and MASK_PATHS have different lengths!" >&2
  exit 1
fi

IDX="$SLURM_ARRAY_TASK_ID"
IMAGE_PATH="${IMAGE_PATHS[$IDX]}"
MASK_PATH="${MASK_PATHS[$IDX]}"

export HYDRA_FULL_ERROR=1

# srun python train_model.py \
#     --config-name=config_2 \
#     image_path="$IMAGE_PATH" \
#     mask_path="$MASK_PATH" \
#     flow_model.model.n_channels=32 \
#     flow_model.cfg.predicts="x_1" \
#     category=small_model_x1

srun python train_model.py \
    --config-name=config_2 \
    image_path="$IMAGE_PATH" \
    mask_path="$MASK_PATH" \
    flow_model.model.n_channels=128 \
    category=big_model
