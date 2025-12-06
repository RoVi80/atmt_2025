#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=2:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_assignment5_timing.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

echo "===== Beam size 1 (greedy) ====="
time python translate.py \
    --cuda \
    --input ./cz-en/data/test_small.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --output cz-en/output_k1.txt \
    --max-len 300 \
    --beam-size 1

echo "===== Beam size 3 ====="
time python translate.py \
    --cuda \
    --input ./cz-en/data/test_small.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --output cz-en/output_k3.txt \
    --max-len 300 \
    --beam-size 3

echo "===== Beam size 5 ====="
time python translate.py \
    --cuda \
    --input ./cz-en/data/test_small.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --output cz-en/output_k5.txt \
    --max-len 300 \
    --beam-size 5
