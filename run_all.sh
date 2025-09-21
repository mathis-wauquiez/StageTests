#!/bin/bash
#SBATCH --job-name=line-int-flow
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --partition=P100,V100,A40,A100
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mathis.wauquiez@eleves.enpc.fr
#SBATCH --array=0-24%12

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

# Model configurations: n_channels, predicts, category
MODEL_CONFIGS=(
    "8 x_0 tiny_model"
    "32 x_0 small_model"
    "32 x_1 small_model_x1"
    "128 x_0 big_model"
    "128 x_1 big_model_x1"
)

if [ "${#IMAGE_PATHS[@]}" -ne "${#MASK_PATHS[@]}" ]; then
  echo "IMAGE_PATHS and MASK_PATHS have different lengths!" >&2
  exit 1
fi

# Calculate which image/mask pair and which model config to use
IDX="$SLURM_ARRAY_TASK_ID"
IMAGE_IDX=$((IDX / ${#MODEL_CONFIGS[@]}))
MODEL_IDX=$((IDX % ${#MODEL_CONFIGS[@]}))

IMAGE_PATH="${IMAGE_PATHS[$IMAGE_IDX]}"
MASK_PATH="${MASK_PATHS[$IMAGE_IDX]}"

# Parse model configuration
MODEL_CONFIG=(${MODEL_CONFIGS[$MODEL_IDX]})
N_CHANNELS="${MODEL_CONFIG[0]}"
PREDICTS="${MODEL_CONFIG[1]}"
CATEGORY="${MODEL_CONFIG[2]}"

export HYDRA_FULL_ERROR=1

# Build the command based on the predicts value
if [ "$PREDICTS" == "x_1" ]; then
    srun python train_model.py \
        --config-name=config_2 \
        image_path="$IMAGE_PATH" \
        mask_path="$MASK_PATH" \
        flow_model.model.n_channels="$N_CHANNELS" \
        flow_model.cfg.predicts="$PREDICTS" \
        category="$CATEGORY"
else
    srun python train_model.py \
        --config-name=config_2 \
        image_path="$IMAGE_PATH" \
        mask_path="$MASK_PATH" \
        flow_model.model.n_channels="$N_CHANNELS" \
        category="$CATEGORY"
fi