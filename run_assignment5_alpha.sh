#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=1:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_assignment5_alpha.out


module load gpu
module load mamba
source activate atmt

INPUT=./cz-en/data/test_small.cz
CKPT=cz-en/checkpoints/checkpoint_best.pt
SRC_TOK=cz-en/tokenizers/cz-bpe-8000.model
TGT_TOK=cz-en/tokenizers/en-bpe-8000.model

echo "===== Beam size 5, alpha = 0.0 (no length penalty) ====="
python translate.py \
    --cuda \
    --input "$INPUT" \
    --src-tokenizer "$SRC_TOK" \
    --tgt-tokenizer "$TGT_TOK" \
    --checkpoint-path "$CKPT" \
    --output cz-en/output_k5_alpha0_0.txt \
    --max-len 300 \
    --beam-size 5 \
    --alpha 0.0

echo "===== Beam size 5, alpha = 0.7 (default) ====="
python translate.py \
    --cuda \
    --input "$INPUT" \
    --src-tokenizer "$SRC_TOK" \
    --tgt-tokenizer "$TGT_TOK" \
    --checkpoint-path "$CKPT" \
    --output cz-en/output_k5_alpha0_7.txt \
    --max-len 300 \
    --beam-size 5 \
    --alpha 0.7

echo "===== Beam size 5, alpha = 1.0 (stronger length normalization) ====="
python translate.py \
    --cuda \
    --input "$INPUT" \
    --src-tokenizer "$SRC_TOK" \
    --tgt-tokenizer "$TGT_TOK" \
    --checkpoint-path "$CKPT" \
    --output cz-en/output_k5_alpha1_0.txt \
    --max-len 300 \
    --beam-size 5 \
    --alpha 1.0
