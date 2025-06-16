#!/bin/sh
#SBATCH --job-name=line-int-flow
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --partition=A100
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mathis.wauquiez@eleves.enpc.fr

export HYDRA_FULL_ERROR=1

# # ---- Small model with multiple images ----
# srun python train_model.py \
#     --config-name=config_2 \
#     data=multiple_images \
#     image_path=data/DaVinciDataset/ \
#     mask_path=data/example_images/mask3.png \
#     flow_model.model.n_channels=32 \
#     category=small_model

# ---- Big model with multiple images ----
srun python train_model.py \
    --config-name=config_2 \
    data/example_images/bricks.jpg \
    mask_path=data/example_images/mask.png \
    flow_model.model.n_channels=128 \
    category=big_model
