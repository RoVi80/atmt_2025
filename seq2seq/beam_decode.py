import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel

def beam_decode(
    model: Seq2SeqModel,
    src_tokens: torch.Tensor,
    src_pad_mask: torch.Tensor,
    max_out_len: int,
    tgt_tokenizer: spm.SentencePieceProcessor,
    beam_size: int,
    device: torch.device,
    length_penalty: float = 0.6
):
    """
    Beam search decoding for Transformer models.
    
    Args:
        model: Trained seq2seq model
        src_tokens: Source sentence tokens [batch_size, src_len]
        src_pad_mask: Source padding mask [batch_size, 1, 1, src_len]
        max_out_len: Maximum output length
        tgt_tokenizer: Target language tokenizer
        beam_size: Number of beams to keep
        device: Device to run on
        length_penalty: Length normalization penalty (0.6 is typical)
    
    Returns:
        List of decoded sequences (one per batch item)
    """
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()
    
    # For simplicity, only support batch_size=1 for beam search
    assert batch_size == 1, "Beam search currently only supports batch_size=1"
    
    # ========== KEY FIX: Encode source ONCE before the loop ==========
    with torch.no_grad():
        encoder_out = model.encode(src_tokens, src_pad_mask)
        # encoder_out: [1, src_len, d_model]
    
    # Expand encoder output for beam_size
    encoder_out_expanded = encoder_out.expand(beam_size, -1, -1)  # [beam_size, src_len, d_model]
    src_pad_mask_expanded = src_pad_mask.expand(beam_size, -1, -1, -1)  # [beam_size, 1, 1, src_len]
    # ==================================================================
    
    # Initialize beams: [beam_size, 1] starting with BOS
    beams = torch.full((beam_size, 1), BOS, dtype=torch.long, device=device)
    beam_scores = torch.zeros(beam_size, device=device)
    beam_scores[1:] = float('-inf')  # Only first beam is active initially
    
    # Track which beams are finished
    finished_beams = []
    finished_scores = []
    
    for step in range(max_out_len):
        # Create target padding mask
        trg_pad_mask = (beams == PAD).unsqueeze(1).unsqueeze(2)  # [beam_size, 1, 1, tgt_len]
        
        # ========== KEY FIX: Use decode() instead of full forward() ==========
        with torch.no_grad():
            output = model.decode(beams, trg_pad_mask, encoder_out_expanded, src_pad_mask_expanded)
            # output: [beam_size, tgt_len, vocab_size]
            
            # Get logits for last position
            next_token_logits = output[:, -1, :]  # [beam_size, vocab_size]
            log_probs = torch.log_softmax(next_token_logits, dim=-1)  # [beam_size, vocab_size]
        # =====================================================================
        
        # Compute scores for all possible next tokens
        vocab_size = log_probs.size(-1)
        
        # Add current beam scores to log probs
        # [beam_size, 1] + [beam_size, vocab_size] = [beam_size, vocab_size]
        candidate_scores = beam_scores.unsqueeze(1) + log_probs
        
        # Flatten to [beam_size * vocab_size] to find top beam_size candidates
        candidate_scores = candidate_scores.view(-1)
        
        # Get top beam_size candidates
        top_scores, top_indices = torch.topk(candidate_scores, beam_size * 2)  # Get extra for finished beams
        
        # Convert flat indices back to (beam_idx, token_idx)
        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size
        
        # Prepare new beams
        new_beams = []
        new_scores = []
        
        for i in range(len(top_scores)):
            beam_idx = beam_indices[i].item()
            token_idx = token_indices[i].item()
            score = top_scores[i].item()
            
            # Create new beam
            new_beam = torch.cat([beams[beam_idx], torch.tensor([token_idx], device=device)])
            
            # Check if this beam ended with EOS
            if token_idx == EOS:
                # Apply length penalty: score / (length ** penalty)
                length = new_beam.size(0)
                normalized_score = score / (length ** length_penalty)
                finished_beams.append(new_beam)
                finished_scores.append(normalized_score)
            else:
                # Keep this beam for next iteration
                if len(new_beams) < beam_size:
                    new_beams.append(new_beam)
                    new_scores.append(score)
            
            # Stop if we have enough beams
            if len(new_beams) >= beam_size:
                break
        
        # If all beams finished, stop
        if len(new_beams) == 0:
            break
        
        # Pad beams to same length
        max_len = max(b.size(0) for b in new_beams)
        beams = torch.stack([
            torch.cat([b, torch.full((max_len - b.size(0),), PAD, dtype=torch.long, device=device)])
            for b in new_beams
        ])
        beam_scores = torch.tensor(new_scores, device=device)
    
    # Select best beam from finished beams
    if finished_beams:
        best_idx = finished_scores.index(max(finished_scores))
        best_beam = finished_beams[best_idx]
    else:
        # If no beam finished with EOS, take the best ongoing beam
        best_idx = beam_scores.argmax().item()
        best_beam = beams[best_idx]
    
    # Remove BOS and PAD, keep up to EOS
    best_beam = best_beam[1:]  # Remove BOS
    best_beam = best_beam.tolist()
    
    # Remove padding and everything after EOS
    if EOS in best_beam:
        eos_idx = best_beam.index(EOS)
        best_beam = best_beam[:eos_idx]  # Don't include EOS
    
    # Remove PAD tokens
    best_beam = [token for token in best_beam if token != PAD]
    
    return [best_beam]  # Return as list for batch compatibility
