#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=pc-er-b8
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --output=logs/%A-%a.out
#SBATCH --error=logs/%A-%a.err
#SBATCH --mail-user=jasony97@cs.ubc.ca
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

set -x
export HF_HOME=~/scratch/trash
export WANDB_API_KEY=d432a06c0e6c4803785e64e780fcd9e9b52d8f4b
cd /ubc/cs/research/plai-scratch/jason/continual-diffusion
conda activate fdm
# export DATA_ROOT=$SLURM_TMPDIR
# export DATASET_DIR=$DATA_ROOT/datasets
# srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c 'mkdir $DATASET_DIR && echo "$(hostname): $(ls -l $DATASET_DIR)" && unzip -q datasets/plaicraft-alex.zip -d $DATASET_DIR'


srun --mpi=pmi2 -n 4 python scripts/video_train.py --dataset=streaming_plaicraft --num_res_blocks=1 --num_channels=128 --num_workers=6 --pad_with_random_frames=False --max_frames=20 --ltm_size=0 --batch_size=8 --n_sample_stm=8 --sample_interval=25000 --save_interval=50000 --lr=1e-4 --weight_decay=1e-5 --diffusion_space=latent --resume_id=oqvzqb3s
# srun --mpi=pmi2 -n 4 python scripts/video_train.py --dataset=streaming_plaicraft --num_res_blocks=1 --num_channels=128 --num_workers=6 --pad_with_random_frames=False --max_frames=20 --ltm_size=20000 --batch_size=8 --n_sample_stm=2 --sample_interval=25000 --save_interval=50000 --lr=1e-4 --weight_decay=1e-5 --diffusion_space=latent --resume_id=wfbmfhok
# srun --mpi=pmi2 -n 4 python scripts/video_train.py --dataset=streaming_plaicraft --num_res_blocks=1 --num_channels=128 --num_workers=6 --pad_with_random_frames=False --max_frames=20 --ltm_size=2000000 --batch_size=8 --n_sample_stm=2 --sample_interval=25000 --save_interval=50000 --lr=1e-4 --weight_decay=1e-5 --diffusion_space=latent --resume_id=1ib38tgy
# srun --mpi=pmi2 -n 4 python scripts/video_train.py --dataset=streaming_plaicraft --num_res_blocks=1 --num_channels=128 --num_workers=6 --pad_with_random_frames=False --max_frames=20 --ltm_size=2000000 --batch_size=8 --n_sample_stm=4 --sample_interval=25000 --save_interval=50000 --lr=1e-4 --weight_decay=1e-5 --diffusion_space=latent
# srun --mpi=pmi2 -n 4 python scripts/video_train.py --dataset=plaicraft --num_res_blocks=1 --num_channels=128 --num_workers=6 --pad_with_random_frames=False --max_frames=20 --batch_size=8 --sample_interval=25000 --save_interval=50000 --lr=1e-4 --weight_decay=1e-5 --diffusion_space=latent --resume_id=0iz7ck2d
