#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_assignment3_joint_bpe_v2_translation.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# Create directories
mkdir -p cz-en/logs_joint_bpe_v2
mkdir -p cz-en/checkpoints_joint_bpe_v2

# PREPARE DATA WITH JOINT BPE
# python preprocess.py \
#    --source-lang cz \
#    --target-lang en \
#    --raw-data ~/shares/cz-en/data/raw \
#    --dest-dir ./cz-en/data/prepared_joint_bpe \
#    --model-dir ./cz-en/tokenizers_joint_bpe \
#    --test-prefix test \
#    --train-prefix train \
#    --valid-prefix valid \
#    --joint-bpe \
#    --joint-vocab-size 16000 \
#    --force-train

# TRAIN WITH JOINT BPE
#python train.py \
#    --cuda \
#    --data cz-en/data/prepared_joint_bpe/ \
#    --src-tokenizer cz-en/tokenizers_joint_bpe/cz-en-joint-bpe-16000.model \
#    --tgt-tokenizer cz-en/tokenizers_joint_bpe/cz-en-joint-bpe-16000.model \
#    --source-lang cz \
#    --target-lang en \
#    --batch-size 64 \
#    --arch transformer \
#    --max-epoch 7 \
#    --log-file cz-en/logs_joint_bpe_v2/train_joint_bpe.log \
#    --save-dir cz-en/checkpoints_joint_bpe_v2/ \
#    --ignore-checkpoints \
#    --encoder-dropout 0.1 \
#    --decoder-dropout 0.1 \
#    --dim-embedding 256 \
#    --attention-heads 4 \
#    --dim-feedforward-encoder 1024 \
#    --dim-feedforward-decoder 1024 \
#    --max-seq-len 300 \
#    --n-encoder-layers 3 \
#    --n-decoder-layers 3


# TRANSLATE
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers_joint_bpe/cz-en-joint-bpe-16000.model \
    --tgt-tokenizer cz-en/tokenizers_joint_bpe/cz-en-joint-bpe-16000.model \
    --checkpoint-path cz-en/checkpoints_joint_bpe_v2/checkpoint_best.pt \
    --output cz-en/output_joint_bpe.txt \
    --max-len 300 \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en

