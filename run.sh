#!/bin/sh
#SBATCH --job-name=int-flow
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --partition=A100
#SBATCH --gpus=3
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mathis.wauquiez@eleves.enpc.fr

export HYDRA_FULL_ERROR=1

srun python train_model.py\
    image_path=data/example_images/grass_big.jpg \
    mask_path=data/example_images/grass_mask.png \
    flow_model.model.n_channels=128 \
    data.train_loader.dataset.sigma=0.05

srun python train_model.py\
    image_path=data/example_images/grass_big.jpg \
    mask_path=data/example_images/grass_mask.png \
    flow_model.model.n_channels=128 \
    data.train_loader.dataset.sigma=0.85