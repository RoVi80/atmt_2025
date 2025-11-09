#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=23:0:0
#SBATCH --gpus=1
#SBATCH --mem=16GB
#SBATCH --output=out_beam_search_testsmall.out

module load gpu
module load mamba
source activate atmt

head -10 ~/shares/cz-en/data/raw/test.cz > ~/test_small.cz
head -10 ~/shares/cz-en/data/raw/test.en > ~/test_small.en

python translate_beam.py \
    --cuda \
    --input ~/test_small.cz \
    --src-tokenizer cz-en/tokenizers_joint_bpe/cz-en-joint-bpe-16000.model \
    --tgt-tokenizer cz-en/tokenizers_joint_bpe/cz-en-joint-bpe-16000.model \
    --checkpoint-path cz-en/checkpoints_joint_bpe_v2/checkpoint_best.pt \
    --output ~/test_beam_small.txt \
    --beam-size 5 \
    --max-len 300 \
    --bleu \
    --reference ~/test_small.en
