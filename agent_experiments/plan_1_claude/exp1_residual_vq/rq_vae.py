"""
Experiment 1: Residual Vector Quantization (RQ-VAE) for Motion Tokenization

MOTIVATION:
The current VQ-VAE uses a single flat codebook of 512 entries to quantize 
motion sequences. This creates a bottleneck: complex motions that differ 
subtly (e.g., signing "hello" vs "hey") may map to the same codes, losing
fine-grained detail that's critical for out-of-domain generalization.

APPROACH:
Residual Vector Quantization (RQ-VAE) decomposes each latent vector through
multiple quantization stages. At each stage, we quantize the residual error
from the previous stage's approximation:
    z_hat = sum_{d=1}^{D} e_{c_d}^{(d)}
where e_{c_d}^{(d)} is the codebook entry at level d, and c_d is the code
index at that level.

This exponentially increases the effective codebook capacity (K^D for K codes
and D levels) while keeping each individual codebook small and well-utilized.

REFERENCES:
- MOGO: Residual Quantized Hierarchical Causal Transformer (2025, arXiv:2506.05952)
  Used MoSA-VQ with residual quantization for motion generation.
- SoundStream: An End-to-End Neural Audio Codec (Zeghidour et al., 2021)
  Pioneered residual VQ for audio compression.
- MoSa: Motion Generation with Scalable Autoregressive Modeling (2024, arXiv:2511.01200)
  Used hierarchical RQ-VAE with multi-scale token preservation.

HYPOTHESIS:
By capturing motion at multiple resolution levels, the LLM can learn coarse
motion patterns (level 1) that generalize across similar sentences, with
fine detail (levels 2-4) providing sentence-specific refinements. This 
coarse-to-fine decomposition should improve out-of-domain generalization
because the base-level patterns are shared across many different signs.

EXPECTED OUTCOMES:
- Higher codebook utilization across all levels (> 90% active codes)
- Lower reconstruction error (especially for hands and face)
- Better generalization to unseen sentences when combined with LLM
- The base codebook should capture semantic motion primitives
"""

import os
import sys
import math
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.resnet import Resnet1D
from models.quantize_cnn import QuantizeEMAReset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.utils import (
    SMPLX_TOTAL_DIM, SMPLX_PARAM_DIMS, set_seeds,
    compute_reconstruction_metrics, compute_codebook_utilization,
    print_experiment_banner, save_experiment_results, EarlyStopping,
)


class ResidualQuantizer(nn.Module):
    """
    Multi-level Residual Vector Quantizer.
    
    At each level d, quantizes the residual from previous levels:
        r_0 = z  (encoder output)
        c_d = argmin_k ||r_{d-1} - e_k^{(d)}||^2
        r_d = r_{d-1} - e_{c_d}^{(d)}
        z_hat = sum_{d=1}^{D} e_{c_d}^{(d)}
    
    Each level uses an independent EMA-updated codebook following
    the QuantizeEMAReset mechanism from T2M-GPT / MotionGPT.
    
    A learnable scale parameter (alpha_d) per level allows the model to
    adaptively weight each residual level's contribution, inspired by
    MOGO's MoSA-VQ (Motion Scale-Adaptive VQ).
    
    References:
        - MOGO (arXiv:2506.05952): MoSA-VQ with learnable scaling
        - SoundStream (Zeghidour et al., 2021): Original residual VQ
    """
    
    def __init__(
        self,
        num_levels: int = 4,
        codebook_size: int = 512,
        code_dim: int = 512,
        mu: float = 0.99,
        use_adaptive_scale: bool = True,
        commitment_weight: float = 0.25,
        level_dropout: float = 0.0,
    ):
        super().__init__()
        
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_weight = commitment_weight
        self.level_dropout = level_dropout
        self.use_adaptive_scale = use_adaptive_scale
        
        # Independent codebook per level
        self.quantizers = nn.ModuleList([
            QuantizeEMAReset(codebook_size, code_dim, mu)
            for _ in range(num_levels)
        ])
        
        # Learnable scale per level (MoSA-VQ inspired)
        if use_adaptive_scale:
            self.level_scales = nn.Parameter(
                torch.ones(num_levels) / num_levels
            )
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, T) -> (N*T, C)"""
        return x.permute(0, 2, 1).contiguous().view(-1, x.shape[1])
    
    def postprocess(self, x_flat: torch.Tensor, N: int, T: int) -> torch.Tensor:
        """(N*T, C) -> (N, C, T)"""
        return x_flat.view(N, T, -1).permute(0, 2, 1).contiguous()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x: (N, C, T) encoder output
        
        Returns:
            x_quantized: (N, C, T) sum of all quantized levels
            total_loss: scalar commitment loss
            info: dict with per-level diagnostics
        """
        N, C, T = x.shape
        x_flat = self.preprocess(x)  # (N*T, C)
        
        residual = x_flat.clone()
        quantized_sum = torch.zeros_like(x_flat)
        total_commit_loss = torch.tensor(0.0, device=x.device)
        
        all_codes = []
        all_perplexities = []
        level_recon_errors = []
        
        # Get scales
        if self.use_adaptive_scale:
            scales = F.softplus(self.level_scales)
        else:
            scales = torch.ones(self.num_levels, device=x.device)
        
        for d in range(self.num_levels):
            quantizer = self.quantizers[d]
            
            # Level dropout: during training, randomly skip higher levels
            # This forces lower levels to carry more information -> better generalization
            if self.training and self.level_dropout > 0 and d > 0:
                if torch.rand(1).item() < self.level_dropout:
                    all_codes.append(torch.zeros(N * T, dtype=torch.long, device=x.device))
                    all_perplexities.append(torch.tensor(0.0))
                    level_recon_errors.append(0.0)
                    continue
            
            # Init codebook if needed (same pattern as QuantizeEMAReset.forward)
            if quantizer.training and not quantizer.init:
                quantizer.init_codebook(residual)
            
            # Quantize residual
            code_idx = quantizer.quantize(residual)
            x_d = quantizer.dequantize(code_idx)
            
            # Update codebook (EMA)
            if quantizer.training:
                perplexity = quantizer.update_codebook(residual, code_idx)
            else:
                perplexity = quantizer.compute_perplexity(code_idx)
            
            # Commitment loss for this level
            commit_loss = F.mse_loss(residual, x_d.detach())
            total_commit_loss = total_commit_loss + self.commitment_weight * commit_loss * scales[d]
            
            # Straight-through estimator
            x_d_st = residual + (x_d - residual).detach()
            
            # Accumulate quantized output with adaptive scale
            quantized_sum = quantized_sum + x_d_st * scales[d]
            
            # Update residual for next level
            residual = residual - x_d.detach()
            
            all_codes.append(code_idx)
            all_perplexities.append(perplexity)
            level_recon_errors.append(residual.pow(2).mean().item())
        
        # Postprocess back to (N, C, T)
        x_quantized = self.postprocess(quantized_sum, N, T)
        
        info = {
            'codes': all_codes,  # List of (N*T,) per level
            'perplexities': all_perplexities,
            'level_recon_errors': level_recon_errors,
            'scales': scales.detach().cpu().tolist() if self.use_adaptive_scale else [1.0] * self.num_levels,
        }
        
        return x_quantized, total_commit_loss, info
    
    def quantize_to_codes(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Encode to multi-level code indices without gradients."""
        N, C, T = x.shape
        x_flat = self.preprocess(x)
        
        residual = x_flat.clone()
        all_codes = []
        
        scales = F.softplus(self.level_scales) if self.use_adaptive_scale else torch.ones(self.num_levels, device=x.device)
        
        for d in range(self.num_levels):
            code_idx = self.quantizers[d].quantize(residual)
            x_d = self.quantizers[d].dequantize(code_idx)
            residual = residual - x_d
            all_codes.append(code_idx.view(N, T))
        
        return all_codes  # List of (N, T) tensors
    
    def decode_from_codes(self, codes: List[torch.Tensor]) -> torch.Tensor:
        """Decode from multi-level code indices."""
        quantized_sum = None
        
        scales = F.softplus(self.level_scales) if self.use_adaptive_scale else torch.ones(self.num_levels, device=codes[0].device)
        
        for d, code_idx in enumerate(codes):
            x_d = self.quantizers[d].dequantize(code_idx.view(-1))
            
            if quantized_sum is None:
                quantized_sum = x_d * scales[d]
            else:
                quantized_sum = quantized_sum + x_d * scales[d]
        
        return quantized_sum


class RQVae(nn.Module):
    """
    Residual Quantized Variational Autoencoder for motion.
    
    Uses the same encoder/decoder architecture as the baseline VQVae
    but replaces the single-level quantizer with multi-level residual
    quantization.
    
    Architecture:
        Encoder: Conv1d -> [DownBlock(Conv+ResNet)] x down_t -> Conv1d
        Quantizer: ResidualQuantizer (multi-level EMA codebooks)
        Decoder: Conv1d -> [UpBlock(ResNet+Upsample+Conv)] x down_t -> Conv1d
    """
    
    def __init__(
        self,
        nfeats: int = 182,
        num_rq_levels: int = 4,
        codebook_size: int = 512,
        code_dim: int = 512,
        output_emb_width: int = 512,
        down_t: int = 3,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = "relu",
        norm: str = None,
        mu: float = 0.99,
        use_adaptive_scale: bool = True,
        commitment_weight: float = 0.25,
        level_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.code_dim = code_dim
        self.num_rq_levels = num_rq_levels
        
        # Encoder (same as baseline)
        self.encoder = self._build_encoder(
            nfeats, output_emb_width, down_t, stride_t,
            width, depth, dilation_growth_rate, activation, norm
        )
        
        # Residual Quantizer
        self.quantizer = ResidualQuantizer(
            num_levels=num_rq_levels,
            codebook_size=codebook_size,
            code_dim=code_dim,
            mu=mu,
            use_adaptive_scale=use_adaptive_scale,
            commitment_weight=commitment_weight,
            level_dropout=level_dropout,
        )
        
        # Decoder (same as baseline)
        self.decoder = self._build_decoder(
            nfeats, output_emb_width, down_t, stride_t,
            width, depth, dilation_growth_rate, activation, norm
        )
    
    def _build_encoder(self, input_emb_width, output_emb_width, down_t, stride_t,
                       width, depth, dilation_growth_rate, activation, norm):
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for _ in range(down_t):
            block = nn.Sequential(
                nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        return nn.Sequential(*blocks)
    
    def _build_decoder(self, input_emb_width, output_emb_width, down_t, stride_t,
                       width, depth, dilation_growth_rate, activation, norm):
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for _ in range(down_t):
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate,
                         reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, width, 3, 1, 1),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        return nn.Sequential(*blocks)
    
    def preprocess(self, x):
        return x.permute(0, 2, 1)  # (B, T, D) -> (B, D, T)
    
    def postprocess(self, x):
        return x.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)
    
    def forward(self, features: torch.Tensor):
        """
        Args:
            features: (B, T, D) motion features
        Returns:
            x_out: (B, T, D) reconstructed features
            loss: commitment loss
            info: per-level diagnostics
        """
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        x_quantized, loss, info = self.quantizer(x_encoder)
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, info
    
    def encode(self, features: torch.Tensor) -> Tuple[List[torch.Tensor], None]:
        """Encode motion to multi-level code indices."""
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        codes = self.quantizer.quantize_to_codes(x_encoder)
        return codes, None
    
    def decode(self, codes: List[torch.Tensor]) -> torch.Tensor:
        """Decode from multi-level code indices."""
        x_d = self.quantizer.decode_from_codes(codes)
        N = codes[0].shape[0]
        T = codes[0].shape[1]
        x_d = x_d.view(N, T, self.code_dim).permute(0, 2, 1).contiguous()
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


def flatten_rq_codes_for_llm(
    codes: List[torch.Tensor],
    codebook_size: int = 512,
    mode: str = "interleaved",
) -> torch.Tensor:
    """
    Flatten multi-level RQ codes into a single sequence for LLM consumption.
    
    Two strategies:
    1. "interleaved": For timestep t, emit [level1_t, level2_t, ..., levelD_t]
       Total sequence length = T * D
       Token offset: level d uses codes in range [d*K, (d+1)*K)
    
    2. "sequential": Emit all level-1 codes, then all level-2, etc.
       Total sequence length = T * D
       Token offset: same as interleaved
    
    The interleaved mode is preferred for autoregressive generation as
    it maintains temporal locality (all information about timestep t is
    contiguous), making it easier for the LLM to learn the relationship
    between coarse and fine tokens at each timestep.
    
    Reference: MOGO uses hierarchical causal structure for similar reasons.
    """
    D = len(codes)
    N, T = codes[0].shape
    
    # Offset codes per level so each level uses a distinct token range
    offset_codes = []
    for d in range(D):
        offset_codes.append(codes[d] + d * codebook_size)
    
    if mode == "interleaved":
        # (N, T, D) then flatten to (N, T*D)
        stacked = torch.stack(offset_codes, dim=2)  # (N, T, D)
        flat = stacked.view(N, T * D)
    elif mode == "sequential":
        flat = torch.cat(offset_codes, dim=1)  # (N, T*D)
    else:
        raise ValueError(f"Unknown flattening mode: {mode}")
    
    return flat


def unflatten_rq_codes_from_llm(
    flat_codes: torch.Tensor,
    num_levels: int = 4,
    codebook_size: int = 512,
    mode: str = "interleaved",
) -> List[torch.Tensor]:
    """Reverse of flatten_rq_codes_for_llm."""
    N = flat_codes.shape[0]
    total_len = flat_codes.shape[1]
    T = total_len // num_levels
    
    if mode == "interleaved":
        reshaped = flat_codes.view(N, T, num_levels)
        codes = []
        for d in range(num_levels):
            level_codes = reshaped[:, :, d] - d * codebook_size
            level_codes = level_codes.clamp(0, codebook_size - 1)
            codes.append(level_codes)
    elif mode == "sequential":
        codes = []
        for d in range(num_levels):
            level_codes = flat_codes[:, d * T:(d + 1) * T] - d * codebook_size
            level_codes = level_codes.clamp(0, codebook_size - 1)
            codes.append(level_codes)
    
    return codes


# ============================================================================
# Training Script
# ============================================================================

def create_rq_vae_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    commit_loss: torch.Tensor,
    loss_weights: torch.Tensor,
    beta: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute weighted reconstruction + commitment loss.
    
    Uses SmoothL1 (Huber) loss per-parameter with higher weights for
    pose, shape, and expression parameters (following the baseline).
    """
    diff = original - reconstructed
    weighted_diff = diff * loss_weights.unsqueeze(0).unsqueeze(0)
    
    recon_loss = F.smooth_l1_loss(weighted_diff, torch.zeros_like(weighted_diff))
    
    total_loss = recon_loss + beta * commit_loss
    
    loss_info = {
        'total': total_loss.item(),
        'recon': recon_loss.item(),
        'commit': commit_loss.item(),
    }
    
    return total_loss, loss_info


def train_rq_vae(
    model: RQVae,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int = 300,
    lr: float = 2e-4,
    beta: float = 0.25,
    output_dir: str = "./exp1_output",
    experiment_name: str = "rq_vae",
):
    """
    Training loop for RQ-VAE.
    
    Includes:
    - Per-parameter weighted loss (pose 10x, shape 5x, expression 8x)
    - Cosine annealing LR scheduler
    - Per-level codebook diagnostics
    - Early stopping on validation loss
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Build loss weights matching the baseline
    param_dims = [10, 63, 45, 45, 3, 10, 3, 3]
    param_starts = np.cumsum([0] + param_dims[:-1]).tolist()
    loss_weights = torch.ones(SMPLX_TOTAL_DIM, device=device)
    loss_weights[param_starts[1]:param_starts[1]+param_dims[1]] = 10.0  # body_pose
    loss_weights[param_starts[2]:param_starts[2]+param_dims[2]] = 10.0  # lhand
    loss_weights[param_starts[3]:param_starts[3]+param_dims[3]] = 10.0  # rhand
    loss_weights[param_starts[0]:param_starts[0]+param_dims[0]] = 5.0   # shape
    loss_weights[param_starts[5]:param_starts[5]+param_dims[5]] = 8.0   # expression
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    early_stop = EarlyStopping(patience=50, min_delta=1e-5)
    
    model.to(device)
    best_val_loss = float('inf')
    history = []
    
    print_experiment_banner("RQ-VAE Training", f"Levels={model.num_rq_levels}, Epochs={epochs}")
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = {'total': 0, 'recon': 0, 'commit': 0}
        epoch_start = time.time()
        
        for batch_idx, batch_data in enumerate(train_loader):
            if isinstance(batch_data, (list, tuple)):
                features = batch_data[0].to(device)
            else:
                features = batch_data.to(device)
            
            x_recon, commit_loss, info = model(features)
            
            total_loss, loss_info = create_rq_vae_loss(
                features, x_recon, commit_loss, loss_weights, beta
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            for k, v in loss_info.items():
                epoch_losses[k] += v
        
        scheduler.step()
        n_batches = len(train_loader)
        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        
        # Validation
        val_loss = None
        if val_loader is not None and (epoch + 1) % 10 == 0:
            model.eval()
            val_total = 0
            val_count = 0
            with torch.no_grad():
                for batch_data in val_loader:
                    if isinstance(batch_data, (list, tuple)):
                        features = batch_data[0].to(device)
                    else:
                        features = batch_data.to(device)
                    
                    x_recon, commit_loss, info = model(features)
                    loss, _ = create_rq_vae_loss(features, x_recon, commit_loss, loss_weights, beta)
                    val_total += loss.item()
                    val_count += 1
            
            val_loss = val_total / max(val_count, 1)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': {
                        'num_rq_levels': model.num_rq_levels,
                        'code_dim': model.code_dim,
                    }
                }, os.path.join(output_dir, 'best_model.pt'))
        
        # Logging
        epoch_time = time.time() - epoch_start
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log_msg = (
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {avg_losses['total']:.6f} (recon: {avg_losses['recon']:.6f}, "
                f"commit: {avg_losses['commit']:.6f}) | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
            if val_loss is not None:
                log_msg += f" | Val: {val_loss:.6f}"
            
            # Per-level perplexities
            if hasattr(info, 'get') or isinstance(info, dict):
                perps = [f"{p.item():.0f}" if torch.is_tensor(p) else f"{p:.0f}" 
                        for p in info.get('perplexities', [])]
                log_msg += f" | Perp: [{', '.join(perps)}]"
            
            print(log_msg)
        
        entry = {
            'epoch': epoch + 1,
            'train_loss': avg_losses['total'],
            'recon_loss': avg_losses['recon'],
            'commit_loss': avg_losses['commit'],
            'lr': scheduler.get_last_lr()[0],
        }
        if val_loss is not None:
            entry['val_loss'] = val_loss
        history.append(entry)
        
        # Early stopping check on validation loss
        if val_loss is not None and early_stop(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'history': history,
    }, os.path.join(output_dir, 'final_model.pt'))
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


def evaluate_rq_vae(
    model: RQVae,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of RQ-VAE including:
    - Reconstruction metrics (per-parameter L1/L2)
    - Per-level codebook utilization
    - Effective codebook capacity
    """
    model.eval()
    all_metrics = defaultdict(list)
    all_codes_per_level = [[] for _ in range(model.num_rq_levels)]
    
    with torch.no_grad():
        for batch_data in data_loader:
            if isinstance(batch_data, (list, tuple)):
                features = batch_data[0].to(device)
            else:
                features = batch_data.to(device)
            
            x_recon, _, info = model(features)
            
            recon_metrics = compute_reconstruction_metrics(features, x_recon)
            for k, v in recon_metrics.items():
                all_metrics[k].append(v)
            
            for d, codes in enumerate(info['codes']):
                all_codes_per_level[d].append(codes.cpu())
    
    # Average reconstruction metrics
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    # Per-level codebook utilization
    codebook_metrics = {}
    for d in range(model.num_rq_levels):
        all_codes = torch.cat(all_codes_per_level[d])
        level_util = compute_codebook_utilization(all_codes, model.quantizer.codebook_size)
        codebook_metrics[f'level_{d}'] = level_util
    
    return {
        'reconstruction': avg_metrics,
        'codebook': codebook_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1: RQ-VAE Training")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with motion .npz files")
    parser.add_argument("--output-dir", type=str, default="./exp1_output")
    parser.add_argument("--num-levels", type=int, default=4, help="Number of RQ levels")
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--code-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level-dropout", type=float, default=0.1)
    args = parser.parse_args()
    
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Config: {vars(args)}")
    
    model = RQVae(
        nfeats=SMPLX_TOTAL_DIM,
        num_rq_levels=args.num_levels,
        codebook_size=args.codebook_size,
        code_dim=args.code_dim,
        level_dropout=args.level_dropout,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\nModel architecture:")
    print(model)
    
    print("\n[NOTE] To train, provide a data directory with motion .npz files.")
    print("       Each .npz should contain 'motion' key with shape (T, 182).")
    print(f"       Example: python rq_vae.py --data-dir /path/to/data --epochs {args.epochs}")
