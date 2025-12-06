import torch
import math
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
    """Beam Search decoding with length penalty."""
    model.eval()
    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()

    # Exercise 3 optimization
    max_len = model.decoder.pos_embed.size(1)

    # GNMT-style length penalty
    def length_penalty(L, alpha):
        if alpha == 0.0:
            return 1.0
        return ((5.0 + L) ** alpha) / ((5.0 + 1.0) ** alpha)

    def norm_score(seq, score):
        L = max(seq.size(1) - 1, 1)  # exclude BOS
        return score / length_penalty(L, alpha)

    beams = [(torch.tensor([[BOS]], device=device), 0.0)]

    for _ in range(max_out_len):
        new_beams = []

        for seq, score in beams:
            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score))
                continue

            with torch.no_grad():
                if seq.size(1) > max_len:
                    seq = seq[:, :max_len]

                trg_pad_mask = (seq == PAD)[:, None, None, :]
                logits = model(src_tokens, src_pad_mask, seq, trg_pad_mask)[:, -1, :]

                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for k in range(beam_size):
                new_token = topk_ids[:, k].unsqueeze(0)
                new_seq = torch.cat([seq, new_token], dim=1)
                new_score = score + topk_log_probs[:, k].item()
                new_beams.append((new_seq, new_score))

        beams = sorted(new_beams,
                       key=lambda x: norm_score(x[0], x[1]),
                       reverse=True)[:beam_size]

        if all(seq[0, -1].item() == EOS for seq, _ in beams):
            break

    best_seq, _ = max(beams, key=lambda x: norm_score(x[0], x[1]))
    return [best_seq.squeeze(0).tolist()]

def beam_search_decode_relative(
    model: Seq2SeqModel,
    src_tokens: torch.Tensor,
    src_pad_mask: torch.Tensor,
    max_out_len: int,
    tgt_tokenizer: spm.SentencePieceProcessor,
    args,
    device: torch.device,
    beam_size: int = 5,
    alpha: float = 0.7,
    rp: float = 0.6,
):
    """
    Beam search with GNMT-style length penalty + Relative Threshold Pruning (rp).
    Implements Eq. (1) in Freitag & Al-Onaizan (2017) in log-prob space.
    """
    model.eval()
    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()

    max_len = model.decoder.pos_embed.size(1)

    def length_penalty(L, alpha):
        if alpha == 0.0:
            return 1.0
        return ((5.0 + L) ** alpha) / ((5.0 + 1.0) ** alpha)

    def norm_score(seq, score):
        L = max(seq.size(1) - 1, 1)  # exclude BOS
        return score / length_penalty(L, alpha)

    # log(rp) for relative pruning in log domain
    log_rp = math.log(rp)

    # (sequence, cumulative_log_score)
    beams = [(torch.tensor([[BOS]], device=device), 0.0)]

    for _ in range(max_out_len):
        new_beams = []

        for seq, score in beams:
            # finished hypothesis – keep as-is
            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score))
                continue

            with torch.no_grad():
                if seq.size(1) > max_len:
                    seq = seq[:, :max_len]

                trg_pad_mask = (seq == PAD)[:, None, None, :]
                logits = model(src_tokens, src_pad_mask, seq, trg_pad_mask)[:, -1, :]

                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for k in range(beam_size):
                new_token = topk_ids[:, k].unsqueeze(0)   # (1, 1)
                new_seq = torch.cat([seq, new_token], dim=1)  # (1, T+1)
                new_score = score + topk_log_probs[:, k].item()
                new_beams.append((new_seq, new_score))

        # --- Relative Threshold Pruning (in log domain) ---
        best_score = max(s for _, s in new_beams)
        threshold = best_score + log_rp
        pruned_beams = [(seq, s) for (seq, s) in new_beams if s >= threshold]

        # safety: never drop *all* beams
        if not pruned_beams:
            pruned_beams = new_beams

        # keep top beam_size by normalized score
        beams = sorted(
            pruned_beams,
            key=lambda x: norm_score(x[0], x[1]),
            reverse=True
        )[:beam_size]

        # EOS early stop (same as before)
        if all(seq[0, -1].item() == EOS for seq, _ in beams):
            break

    best_seq, _ = max(beams, key=lambda x: norm_score(x[0], x[1]))
    return [best_seq.squeeze(0).tolist()]


def beam_search_decode_absolute(
    model: Seq2SeqModel,
    src_tokens: torch.Tensor,
    src_pad_mask: torch.Tensor,
    max_out_len: int,
    tgt_tokenizer: spm.SentencePieceProcessor,
    args,
    device: torch.device,
    beam_size: int = 5,
    alpha: float = 0.7,
    ap: float = 2.5,
):
    """
    Beam search with GNMT-style length penalty + Absolute Threshold Pruning (ap).
    Implements Eq. (2) in Freitag & Al-Onaizan (2017) in log-prob space.
    """
    model.eval()
    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()

    max_len = model.decoder.pos_embed.size(1)

    def length_penalty(L, alpha):
        if alpha == 0.0:
            return 1.0
        return ((5.0 + L) ** alpha) / ((5.0 + 1.0) ** alpha)

    def norm_score(seq, score):
        L = max(seq.size(1) - 1, 1)
        return score / length_penalty(L, alpha)

    beams = [(torch.tensor([[BOS]], device=device), 0.0)]

    for _ in range(max_out_len):
        new_beams = []

        for seq, score in beams:
            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score))
                continue

            with torch.no_grad():
                if seq.size(1) > max_len:
                    seq = seq[:, :max_len]

                trg_pad_mask = (seq == PAD)[:, None, None, :]
                logits = model(src_tokens, src_pad_mask, seq, trg_pad_mask)[:, -1, :]

                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for k in range(beam_size):
                new_token = topk_ids[:, k].unsqueeze(0)
                new_seq = torch.cat([seq, new_token], dim=1)
                new_score = score + topk_log_probs[:, k].item()
                new_beams.append((new_seq, new_score))

        # --- Absolute Threshold Pruning in log domain ---
        best_score = max(s for _, s in new_beams)
        threshold = best_score - ap
        pruned_beams = [(seq, s) for (seq, s) in new_beams if s >= threshold]

        if not pruned_beams:
            pruned_beams = new_beams

        beams = sorted(
            pruned_beams,
            key=lambda x: norm_score(x[0], x[1]),
            reverse=True
        )[:beam_size]

        if all(seq[0, -1].item() == EOS for seq, _ in beams):
            break

    best_seq, _ = max(beams, key=lambda x: norm_score(x[0], x[1]))
    return [best_seq.squeeze(0).tolist()]
