import os
import logging
import argparse
import time
import sacrebleu
from tqdm import tqdm

import torch
import sentencepiece as spm
from torch.serialization import default_restore_location

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from seq2seq.beam_decode import beam_decode
from seq2seq import models, utils

def get_args():
    """ Defines generation-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model with Beam Search')
    parser.add_argument('--cuda', action='store_true', help='Use a GPU')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')

    # Add data arguments
    parser.add_argument('--input', required=True, help='Path to the raw text file to translate')
    parser.add_argument('--src-tokenizer', help='path to source sentencepiece tokenizer', required=True)
    parser.add_argument('--tgt-tokenizer', help='path to target sentencepiece tokenizer', required=True)
    parser.add_argument('--checkpoint-path', required=True, help='path to the model file')
    parser.add_argument('--output', required=True, type=str, help='path to the output file')
    parser.add_argument('--max-len', default=300, type=int, help='maximum length of generated sequence')
    
    # Beam search parameters
    parser.add_argument('--beam-size', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--length-penalty', default=0.6, type=float, help='length penalty for beam search')
    
    # BLEU computation arguments
    parser.add_argument('--bleu', action='store_true', help='If set, compute BLEU score after translation')
    parser.add_argument('--reference', type=str, help='Path to the reference file')
    
    return parser.parse_args()


def main(args):
    """ Main translation function with beam search """
    torch.manual_seed(args.seed)
    
    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint_path}...')
    state_dict = torch.load(args.checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'), weights_only=False)
    
    # Get model args from checkpoint for building the model
    model_args = state_dict['args']
    
    # Load tokenizers
    print('Loading tokenizers...')
    src_tokenizer = utils.load_tokenizer(args.src_tokenizer)
    tgt_tokenizer = utils.load_tokenizer(args.tgt_tokenizer)

    # Read input sentences
    print(f'Reading input from {args.input}...')
    with open(args.input, encoding="utf-8") as f:
        src_lines = [line.strip() for line in f if line.strip()]

    print(f"Translating {len(src_lines)} sentences with beam size {args.beam_size}...")

    # Encode input sentences
    src_encoded = [torch.tensor(src_tokenizer.Encode(line, out_type=int)) for line in src_lines]
    src_encoded = [s if len(s) <= args.max_len else s[:args.max_len] for s in src_encoded]

    # Build model using model_args from checkpoint
    print('Building model...')
    model = models.build_model(model_args, src_tokenizer, tgt_tokenizer)
    if args.cuda:
        model = model.cuda()
    model.eval()
    model.load_state_dict(state_dict['model'])
    print(f'Loaded model from checkpoint {args.checkpoint_path}')

    DEVICE = 'cuda' if args.cuda else 'cpu'
    PAD = src_tokenizer.pad_id()

    # Clear output file
    print(f'Writing translations to {args.output}...')
    with open(args.output, 'w', encoding="utf-8") as out_file:
        out_file.write('')

    translations = []
    start_time = time.perf_counter()

    # Translate sentence by sentence (beam search works on single sentences)
    for src_tokens in tqdm(src_encoded, desc="Translating"):
        # Add batch dimension and move to device
        src_tokens = src_tokens.unsqueeze(0).to(DEVICE)  # [1, src_len]
        
        # Create source padding mask
        src_pad_mask = (src_tokens == PAD).unsqueeze(1).unsqueeze(2)  # [1, 1, 1, src_len]
        
        with torch.no_grad():
            # Decode with beam search
            prediction = beam_decode(
                model=model,
                src_tokens=src_tokens,
                src_pad_mask=src_pad_mask,
                max_out_len=args.max_len,
                tgt_tokenizer=tgt_tokenizer,
                beam_size=args.beam_size,
                device=DEVICE,
                length_penalty=args.length_penalty
            )
        
        # Decode to string
        translation = tgt_tokenizer.Decode(prediction[0])
        translations.append(translation)
        
        # Write to file
        with open(args.output, 'a', encoding="utf-8") as out_file:
            out_file.write(translation + '\n')
    
    end_time = time.perf_counter()
    print(f'\nWrote {len(translations)} lines to {args.output}')
    print(f'Translation completed in {end_time - start_time:.2f} seconds')

    # Compute BLEU score if requested
    if args.bleu and args.reference:
        print(f'Computing BLEU score...')
        with open(args.reference, encoding='utf-8') as ref_file:
            references = [line.strip() for line in ref_file if line.strip()]
        
        if len(references) != len(translations):
            print(f"Warning: Reference ({len(references)}) and hypothesis ({len(translations)}) counts don't match")
        
        bleu = sacrebleu.corpus_bleu(translations, [references])
        print(f"BLEU score: {bleu.score:.2f}")
        return bleu.score
    
    return None


if __name__ == '__main__':
    args = get_args()
    main(args)
