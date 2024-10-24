#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=sample
#SBATCH --gres=gpu:1
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


WANDB_IDS=(oqvzqb3s wfbmfhok 1ib38tgy 0iz7ck2d)
NICKNAMES=(online er er-full offline)
export WANDB_ID=${WANDB_IDS[$SLURM_ARRAY_TASK_ID]}

export EPOCH=1800000
export MODEL_NAME=ema_0.9999_${EPOCH}
export SAMPLER="heun-80-inf-0-1-1000-0.002-7-50"
export COLLECT_OUTPUT_DIR="plaicraft/comparison_test_${EPOCH}"

NUM_GENERATIONS=500
MAX_FRAMES=20
MAX_LATENT_FRAMES=10
VID_LENGTH=50
N_START_FRAMES=10
SAMPLE_SCHEME=autoreg
EVAL_ON_TRAIN=False
DECODE_CHUNK_SIZE=3
SAMPLE_BATCH_SIZE=1
FVD_BATCH_SIZE=10
N_VISUALIZE=4
EVAL_DATASET_CONFIG='default'
UPPER_FRAME_RANGE=500000

if [ "$EVAL_ON_TRAIN" = True ]; then
  STREAM_TYPE='train'
else
  STREAM_TYPE='test'
fi

CHECKPOINT_PATH=checkpoints/${WANDB_ID}/${MODEL_NAME}.pt
RESULT_PATH=results/${WANDB_ID}/${MODEL_NAME}_${SAMPLER}/${SAMPLE_SCHEME}_${MAX_FRAMES}_${VID_LENGTH}_${N_START_FRAMES}_${EVAL_DATASET_CONFIG}_0_${UPPER_FRAME_RANGE}_${STREAM_TYPE}
COLLECT_PREFIX=${MODEL_NAME}_${SAMPLER}/${SAMPLE_SCHEME}_${MAX_FRAMES}_${VID_LENGTH}_${N_START_FRAMES}_${EVAL_DATASET_CONFIG}_0_${UPPER_FRAME_RANGE}_${STREAM_TYPE}

python scripts/video_sample.py $CHECKPOINT_PATH --T=$VID_LENGTH --stop_index=$NUM_GENERATIONS --num_sampled_videos=$NUM_GENERATIONS --max_frames=$MAX_FRAMES --n_obs=$N_START_FRAMES --sampling_scheme=$SAMPLE_SCHEME --batch_size=$SAMPLE_BATCH_SIZE --eval_on_train=$EVAL_ON_TRAIN --sampler=$SAMPLER --eval_dataset_config=$EVAL_DATASET_CONFIG --decode_chunk_size=$DECODE_CHUNK_SIZE --upper_frame_range=$UPPER_FRAME_RANGE
python scripts/video_make_mp4.py --eval_dir=$RESULT_PATH --do_n=$N_VISUALIZE --add_gt=True --num_sampled_videos=$NUM_GENERATIONS --decode_chunk_size=$DECODE_CHUNK_SIZE
python scripts/video_fvd.py --eval_dir=$RESULT_PATH --num_videos=$NUM_GENERATIONS --batch_size=$FVD_BATCH_SIZE --decode_chunk_size=$DECODE_CHUNK_SIZE
python scripts/collect_results.py --wandb_ids ${WANDB_IDS[@]} --nicknames ${NICKNAMES[@]} --txt_paths ${COLLECT_PREFIX}/fvd-${NUM_GENERATIONS}-0.txt --video_path ${COLLECT_PREFIX}/videos/${N_VISUALIZE}_1.gif --output_dir $COLLECT_OUTPUT_DIR
