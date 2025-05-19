#!/bin/sh
#SBATCH --job-name=train_model
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --partition=A100
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mathis.wauquiez@eleves.enpc.fr

conda activate stage

srun python train_model.py \
    image_path=data/example_images/2.png \
    mask_path=data/example_images/mask3.png
