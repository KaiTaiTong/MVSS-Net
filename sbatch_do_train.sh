#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=def-panos
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=48:00:00

module load StdEnv/2020
module load gentoo/2020
module load python/3.7.7
module load intel/2020.1.217  cuda/11.4

# Sample for MVSS-Net, similar to MantraNet
source ~/projects/def-panos/EECE571_2022/FakeImageDetection/env/MVSS-Net/bin/activate
cd ~/scratch/MVSS_net
# tensorboard --logdir=logs --host 0.0.0.0
# python -u train.py --load_path ./ckpt/mvssnet_casia.pt --batch_size 8
python -u train.py --batch_size 12 --n_epochs 100 --image_size 128 --lambda_seg 0.2 --lambda_clf 0.01

echo "End"
