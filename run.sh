#!/bin/sh
#SBATCH --job-name=train_model
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --partition=A100
#SBATCH --gpus=3
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=48GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mathis.wauquiez@eleves.enpc.fr

conda init
conda activate flow_matching

srun python train_model.py --multirun \
    image_path=data/example_images/2.png \
    mask_path=data/example_images/mask3.png \
    data.train_loader.dataset.sigma=0,0.2,0.4,0.6,0.8,1.0