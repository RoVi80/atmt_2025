import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel

def decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
           tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device):
    """Decodes a sequence without teacher forcing. Works by relying on the model's own predictions, rather than the ground truth (trg_)"""
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()

    # Exercise 3: compute max_len once outside the loop
    max_len = model.decoder.pos_embed.size(1)

    generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for t in range(max_out_len):
        # Use precomputed max_len instead of recomputing it each step
        if generated.size(1) > max_len:
            generated = generated[:, :max_len]

        # Ensure trg_pad_mask has shape (batch_size, seq_len)
        trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)

        # Forward pass: use only the generated tokens so far
        with torch.no_grad():
            output = model(src_tokens, src_pad_mask, generated, trg_pad_mask)

        # Get the logits for the last time step
        next_token_logits = output[:, -1, :]  # last time step
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy

        # Append next token to each sequence
        generated = torch.cat([generated, next_tokens], dim=1)

        # Mark sequences as finished if EOS is generated
        finished = finished | (next_tokens.squeeze(1) == EOS)
        if finished.all():
            break

    # Remove initial BOS token and anything after EOS
    predicted_tokens = []
    for seq in generated[:, 1:].tolist():
        if EOS in seq:
            idx = seq.index(EOS)
            seq = seq[:idx+1]
        predicted_tokens.append(seq)
    return predicted_tokens


def beam_search_decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
                       tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device,
                       beam_size: int = 5, alpha: float = 0.7):
    """Beam Search decoding compatible with Transformer-based Seq2Seq models."""
    model.eval()
    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()

    # Exercise 3: compute max_len once outside the time-step loop
    max_len = model.decoder.pos_embed.size(1)

    # Helper: length-penalized score (Google NMT-style)
    def length_penalty(seq_len: int, alpha: float) -> float:
        """
        Compute length penalty given sequence length (excluding BOS) and alpha.
        lp(L) = ((5 + L)^alpha) / ((5 + 1)^alpha)
        """
        if alpha == 0.0:
            return 1.0
        return ((5.0 + seq_len) ** alpha) / ((5.0 + 1.0) ** alpha)

    def scored_key(seq, score):
        """
        Return the normalized score used to rank beams.
        seq has shape (1, T). We define length as T-1 (exclude BOS).
        """
        seq_len = seq.size(1) - 1  # ignore BOS
        lp = length_penalty(max(seq_len, 1), alpha)
        return score / lp

    # __QUESTION 1: what does this line set up and why is the beam represented this way?
    beams = [(torch.tensor([[BOS]], device=device), 0.0)]  # (sequence, cumulative_log_score)

    for _ in range(max_out_len):
        new_beams = []
        for seq, score in beams:
            # If this hypothesis already ended with EOS, keep it as-is
            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score))
                continue

            with torch.no_grad():
                # Use precomputed max_len
                if seq.size(1) > max_len:
                    seq = seq[:, :max_len]

                # __QUESTION 2: Why do we need to create trg_pad_mask here and how does it affect the model's predictions?
                trg_pad_mask = (seq == PAD)[:, None, None, :]
                logits = model(src_tokens, src_pad_mask, seq, trg_pad_mask)[:, -1, :]

                # __QUESTION 3: Explain the purpose of applying log_softmax and selecting top-k tokens here.
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for k in range(beam_size):
                # __QUESTION 4: explain the tensor shapes and the logic when creating new_seq and new_score below. Is any broadcasting or indexing issue possible?
                new_token = topk_ids[:, k].unsqueeze(0)   # shape (1, 1)
                new_seq = torch.cat([seq, new_token], dim=1)  # (1, T+1)
                new_score = score + topk_log_probs[:, k].item()  # still raw sum of log-probs
                new_beams.append((new_seq, new_score))

        # Rank beams by length-normalized score and keep the top 'beam_size'
        beams = sorted(new_beams,
                       key=lambda x: scored_key(x[0], x[1]),
                       reverse=True)[:beam_size]

        # __QUESTION 5: Why do we check for EOS here and what does it imply for beam search?
        if all(seq[0, -1].item() == EOS for seq, _ in beams):
            break

    # Pick the best beam according to the same length-normalized score
    best_seq, best_score = max(beams, key=lambda x: scored_key(x[0], x[1]))
    # __QUESTION 6: What is returned, and why are we squeezing, converting to list and wrapping in another list here?
    return [best_seq.squeeze(0).tolist()]
