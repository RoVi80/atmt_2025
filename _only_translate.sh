#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=2:35:00
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=translate_only_out.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd "$HOME/data/atmt_2025"
mkdir -p cz-en/output

python translate.py \
  --cuda \
  --input       ~/shares/cz-en/data/raw/test.cz \
  --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
  --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
  --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
  --output cz-en/output/output.txt \
  --max-len 300 \
  --batch-size 64 \
  --bleu \
  --reference  ~/shares/cz-en/data/raw/test.en
