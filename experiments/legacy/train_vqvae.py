import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import warnings
import json
from datetime import datetime
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
from tqdm import tqdm

# ==============================================================================
# 0) SETUP: Architecture files
# ==============================================================================
# Make sure your mGPT folder is in the Python path
# sys.path.append('/path/to/your/mGPT_folder')
from mGPT.archs.mgpt_vq import VQVae
from mGPT.archs.tools import quantize_cnn

warnings.filterwarnings("ignore")

# ==============================================================================
# 1) CONFIGURATION
# ==============================================================================
SANITY_CHECK_ENABLED = True
sanity_check_counter = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", DEVICE)
print(f"Sanity checks are {'ENABLED' if SANITY_CHECK_ENABLED else 'DISABLED'}.")

# ==============================================================================
# 2) VQ-VAE MODEL (Your instrumented classes are fine)
# ==============================================================================
class QuantizeEMAReset_Sanity(quantize_cnn.QuantizeEMAReset):
    def forward(self, x, current_batch_idx=0):
        global sanity_check_counter
        N, width, T = x.shape
        x_proc = self.preprocess(x)
        if SANITY_CHECK_ENABLED and current_batch_idx == 0 and sanity_check_counter == 0:
            print("[Quantizer.forward] Input shape `x`: ", x.shape)
            print("[Quantizer.forward] Shape after preprocess `x_proc`: ", x_proc.shape)
            print(f"[Quantizer.forward] Codebook shape: {self.codebook.shape}")
            if self.training and not self.init: print("[Quantizer.forward] Codebook is UNINITIALIZED.")
            else: print(f"[Quantizer.forward] Codebook stats: min={self.codebook.min():.3f}, max={self.codebook.max():.3f}, mean={self.codebook.mean():.3f}")
        if self.training and not self.init: self.init_codebook(x_proc)
        code_idx = self.quantize(x_proc)
        x_d = self.dequantize(code_idx)
        if SANITY_CHECK_ENABLED and current_batch_idx == 0 and sanity_check_counter == 0:
            print(f"[Quantizer.forward] Code index range: min={code_idx.min()}, max={code_idx.max()}")
            assert code_idx.max() < self.nb_code, "A code index is out of bounds!"
        if self.training: perplexity = self.update_codebook(x_proc, code_idx)
        else: perplexity = self.compute_perplexity(code_idx)
        commit_loss = F.mse_loss(x_proc, x_d.detach())
        x_d = x_proc + (x_d - x_proc).detach()
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        return x_d, commit_loss, perplexity

class VQVae_Sanity(VQVae):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.quantizer, quantize_cnn.QuantizeEMAReset):
            self.quantizer = QuantizeEMAReset_Sanity(
                self.quantizer.nb_code, self.quantizer.code_dim, self.quantizer.mu
            )
    def forward(self, features, current_batch_idx=0):
        global sanity_check_counter
        x_in = self.preprocess(features)
        if SANITY_CHECK_ENABLED and current_batch_idx == 0 and sanity_check_counter == 0: print("[VQVae.forward] Shape after preprocess (permute): ", x_in.shape)
        x_encoder = self.encoder(x_in)
        if SANITY_CHECK_ENABLED and current_batch_idx == 0 and sanity_check_counter == 0:
            print("[VQVae.forward] Shape after encoder `x_encoder`: ", x_encoder.shape)
            total_downsample_factor = 2**3
            expected_len = math.ceil(features.shape[1] / total_downsample_factor)
            print(f"[VQVae.forward] Calculated expected quantized length: ~{expected_len}")
            assert abs(x_encoder.shape[2] - expected_len) <= 1, "Temporal downsampling seems incorrect."
        x_quantized, loss, perplexity = self.quantizer(x_encoder, current_batch_idx)
        if SANITY_CHECK_ENABLED and current_batch_idx == 0 and sanity_check_counter == 0: print("[VQVae.forward] Shape after quantizer `x_quantized`: ", x_quantized.shape)
        x_decoder = self.decoder(x_quantized)
        if SANITY_CHECK_ENABLED and current_batch_idx == 0 and sanity_check_counter == 0:
            print("[VQVae.forward] Shape after decoder `x_decoder`: ", x_decoder.shape)
            assert x_decoder.shape[2] == features.shape[1], "Decoder output temporal dim mismatch!"
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity

# Monkey-patching
sys.modules['mGPT.archs.mgpt_vq'].VQVae = VQVae_Sanity
sys.modules['mGPT.archs.mgpt_vq'].QuantizeEMAReset = QuantizeEMAReset_Sanity

class MotionGPT_VQVAE_Wrapper(nn.Module):
    def __init__(self, smpl_dim, codebook_size=512, code_dim=512, **kwargs):
        super().__init__()
        self.smpl_dim = smpl_dim
        self.vqvae = VQVae(
            nfeats=smpl_dim, code_num=codebook_size, code_dim=code_dim,
            output_emb_width=code_dim, **kwargs
        )
        param_dims = [10, 63, 45, 45, 3, 10, 3, 3]
        param_starts = np.cumsum([0] + param_dims[:-1]).tolist()
        loss_weights = torch.ones(smpl_dim)
        loss_weights[param_starts[1]:param_starts[5]] = 10.0
        loss_weights[param_starts[0]:param_starts[1]] = 5.0
        loss_weights[param_starts[5]:param_starts[6]] = 8.0
        self.register_buffer('loss_weights', loss_weights)
        print(f"Initialized MotionGPT VQ-VAE with {codebook_size} codebook size")
    def forward(self, x, current_batch_idx=0):
        global sanity_check_counter
        if SANITY_CHECK_ENABLED and current_batch_idx == 0 and sanity_check_counter == 0:
            print("\n" + "="*50)
            print("--- VQ-VAE WRAPPER SANITY CHECK (Batch 0) ---")
            print(f"[Input] Shape of input features `x`: {x.shape}")
            print("-"*50)
        x_recon, vq_loss, perplexity = self.vqvae(x, current_batch_idx)
        if SANITY_CHECK_ENABLED and current_batch_idx == 0 and sanity_check_counter == 0:
            print("[Output] Shape of reconstructed features `x_recon`: ", x_recon.shape)
            assert x.shape == x_recon.shape, "Shape mismatch!"
            print(f"[Output] vq_loss: {vq_loss.item():.6f}, perplexity: {perplexity.item():.2f}")
            print("--- VQ-VAE WRAPPER SANITY CHECK COMPLETE ---")
            print("="*50 + "\n")
        indices, _ = self.vqvae.encode(x)
        return x_recon, vq_loss, indices, perplexity

# ==============================================================================
# 3) DATA LOADING
# ==============================================================================
def load_motion_from_npz(file_path):
    try:
        with np.load(file_path) as data:
            motion_data = data['motion']
            return torch.tensor(motion_data, dtype=torch.float32)
    except Exception as e:
        print(f"Warning: Could not load {os.path.basename(file_path)}. Skipping. Error: {e}")
        return None

class NpzMotionDataset(Dataset):
    def __init__(self, root_dir, stats_path=None, min_seq_len=64):
        self.min_seq_len = min_seq_len
        print(f"\n[Dataset] Initializing from NPZ files in: '{root_dir}'")
        glob_pattern = os.path.join(root_dir, '**', '*.npz')
        self.files = glob.glob(glob_pattern, recursive=True)
        if not self.files:
            raise FileNotFoundError(f"FATAL: No .npz files found at '{glob_pattern}'.")
        print(f"[Dataset] Found {len(self.files)} total .npz files.")

        if stats_path and os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location='cpu')
            self.mean = stats['mean']
            self.std = stats['std']
            print("[Dataset] Successfully loaded normalization stats to CPU.")
        else:
            print("❗ [Dataset] WARNING: Stats file not found. Proceeding without normalization. This will affect loss values and model performance.")
            self.mean = 0
            self.std = 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        seq = load_motion_from_npz(file_path)
        if seq is None or seq.shape[0] < self.min_seq_len:
            return None
        normalized_seq = (seq - self.mean) / self.std
        return normalized_seq

# ==============================================================================
# 4) CHECKPOINT & CODEBOOK INITIALIZATION
# ==============================================================================
class CheckpointManager:
    # (Your CheckpointManager code is fine, no changes needed here)
    def __init__(self, checkpoint_dir, max_checkpoints=3):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
    def save_checkpoint(self, model, optimizer, epoch, loss, metadata=None):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'vqvae_epoch_{epoch:03d}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }, checkpoint_path)
        print(f"✅ Saved checkpoint: {checkpoint_path}")
        self.cleanup_old_checkpoints()
    def cleanup_old_checkpoints(self):
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'vqvae_epoch_*.pt'))
        if len(checkpoints) > self.max_checkpoints:
            checkpoints.sort(key=os.path.getmtime)
            for old_checkpoint in checkpoints[:-self.max_checkpoints]:
                os.remove(old_checkpoint)
                print(f"🗑️ Removed old checkpoint: {old_checkpoint}")
    def load_latest_checkpoint(self):
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'vqvae_epoch_*.pt'))
        if not checkpoints: return None
        latest_checkpoint_path = max(checkpoints, key=os.path.getmtime)
        print(f"🔄 Loading latest checkpoint: {latest_checkpoint_path}")
        return torch.load(latest_checkpoint_path, map_location=DEVICE, weights_only=False)

def initialize_codebook_from_dataset(model, dataloader, num_batches=100):
    print(f"⚙️ Collecting data from {num_batches} batches for codebook initialization...")
    all_latents = []
    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            if i >= num_batches: break
            if batch_data and batch_data[0] is not None:
                motion_batch, _ = batch_data
                x = motion_batch.to(DEVICE)
                z_e = model.vqvae.encoder(model.vqvae.preprocess(x))
                z_e_flat = z_e.permute(0, 2, 1).reshape(-1, z_e.shape[1])
                all_latents.append(z_e_flat.cpu())
    if not all_latents: raise ValueError("Could not collect any latents for initialization.")
    all_latents = torch.cat(all_latents, dim=0)
    print(f"Collected {all_latents.shape[0]} latent vectors.")
    codebook_size = model.vqvae.quantizer.nb_code
    indices = torch.randperm(all_latents.shape[0])[:codebook_size]
    initial_codebook = all_latents[indices].to(DEVICE)
    model.vqvae.quantizer.init_codebook(initial_codebook)
    print("✅ Codebook initialized successfully from a diverse data sample.")
    model.train()

# ==============================================================================
# 5) CORRECTED & COMPLETE TRAINING FUNCTION (No Globals)
# ==============================================================================
def train_vqvae_colab(vq_model, dataset, checkpoint_dir, num_epochs=300, batch_size=32, lr=2e-4):
    """
    The complete, updated training function for Colab using .npz files.
    This version avoids global variables by accepting checkpoint_dir as an argument.
    """
    global sanity_check_counter
    print("\n" + "="*70 + "\n     STARTING VQ-VAE TRAINING ON COLAB     \n" + "="*70)

    optimizer = torch.optim.AdamW(vq_model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn = nn.SmoothL1Loss(reduction='none')
    # Use the passed-in checkpoint_dir
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    start_epoch = 1
    checkpoint = checkpoint_manager.load_latest_checkpoint()
    if checkpoint:
        vq_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 1) + 1
        print(f"✅ Resumed training from epoch {start_epoch}")
    else: print("No CheckPoint Found")
    vq_model.to(DEVICE).train()
    codebook_size = vq_model.vqvae.quantizer.nb_code

    def collate_fn_enhanced(batch):
        batch = [item for item in batch if item is not None]
        if not batch: return None, None
        batch.sort(key=lambda x: x.shape[0], reverse=True)
        max_len = min(batch[0].shape[0], 256)
        padded_max_len = math.ceil(max_len / 8) * 8
        padded_batch = torch.zeros(len(batch), padded_max_len, batch[0].shape[1])
        lengths = [min(x.shape[0], padded_max_len) for x in batch]
        for i, x_item in enumerate(batch):
            padded_batch[i, :lengths[i], :] = x_item[:lengths[i], :]
        return padded_batch, torch.tensor(lengths)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                            collate_fn=collate_fn_enhanced, drop_last=True, pin_memory=True)

    if start_epoch == 1 and not getattr(vq_model.vqvae.quantizer, 'init', False):
        initialize_codebook_from_dataset(vq_model, dataloader, num_batches=100)

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'='*30} EPOCH {epoch}/{num_epochs} {'='*30}")
        epoch_losses, epoch_vq_losses, epoch_rec_losses, epoch_perplexity = [], [], [], []
        epoch_indices = []

        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            if not batch_data or batch_data[0] is None: continue

            motion_batch, lengths = batch_data
            x = motion_batch.to(DEVICE)
            x_recon, vq_loss, indices, perplexity = vq_model(x, batch_idx)

            rec_loss_unreduced = loss_fn(x_recon, x) * vq_model.loss_weights
            mask = torch.zeros_like(x[:, :, 0], device=DEVICE)
            for i, length in enumerate(lengths): mask[i, :length] = 1.0
            mask = mask.unsqueeze(-1).expand_as(rec_loss_unreduced)
            rec_loss = (rec_loss_unreduced * mask).sum() / mask.sum()

            # vq_weight = max(150.0 * (0.97 ** max(0, epoch - 3)), 1.0)
            beta = 0.25 # This is a standard and effective value.
            total_loss = rec_loss + (beta * vq_loss)
            # total_loss = rec_loss + (vq_weight * vq_loss)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vq_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(total_loss.item())
            epoch_vq_losses.append(vq_loss.item())
            epoch_rec_losses.append(rec_loss.item())
            epoch_perplexity.append(perplexity.item())
            epoch_indices.append(indices.cpu().numpy().flatten())

            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f"\n[E:{epoch:03d}] B:{batch_idx:03d} | Loss: {total_loss.item():.4f} (Rec: {rec_loss.item():.4f}, VQ: {vq_loss.item():.6f}) | Perplexity: {perplexity.item():.2f}")

            if SANITY_CHECK_ENABLED and batch_idx == 0 and sanity_check_counter == 0:
                sanity_check_counter += 1

        if not epoch_losses: continue

        all_epoch_indices_flat = np.concatenate(epoch_indices)
        counts = np.bincount(all_epoch_indices_flat, minlength=codebook_size)
        avg_usage = (counts > 0).sum()
        with torch.no_grad(): code_variance = vq_model.vqvae.quantizer.codebook.var(dim=0).mean().item()

        print(f"\n[EPOCH {epoch:03d} SUMMARY]")
        print(f"  Avg Loss: {np.mean(epoch_losses):.4f} (Rec: {np.mean(epoch_rec_losses):.4f}, VQ: {np.mean(epoch_vq_losses):.6f})")
        print(f"  Avg Perplexity: {np.mean(epoch_perplexity):.2f}")
        print(f"  Codebook Usage: {avg_usage}/{codebook_size} ({(avg_usage/codebook_size)*100:.1f}%) | Variance: {code_variance:.6f}")

        # Use the passed-in checkpoint_dir for saving plots
        hist_path = os.path.join(checkpoint_dir, f'codebook_usage_epoch_{epoch:03d}.png')
        plt.figure(figsize=(12, 6)); plt.hist(all_epoch_indices_flat, bins=codebook_size); plt.title(f'Codebook Usage - Epoch {epoch}'); plt.savefig(hist_path); plt.close()

        if epoch > 0 and epoch % 5 == 0:
            print("\n--- Performing End-of-Epoch Tasks ---")
            vq_model.eval()
            with torch.no_grad():
                val_data = next(iter(dataloader))
                if val_data and val_data[0] is not None:
                    motion_batch, lengths = val_data
                    x_val = motion_batch.to(DEVICE)
                    x_recon_val, _, _, _ = vq_model(x_val, -1)
                    orig = x_val[0, :lengths[0]].cpu().numpy()
                    recon = x_recon_val[0, :lengths[0]].cpu().numpy()
                    mse = ((orig - recon) ** 2).mean()
                    print(f"Reconstruction MSE on sample: {mse:.6f}")

            with torch.no_grad():
                usage_threshold = 10
                underutilized_indices = torch.from_numpy(np.where(counts < usage_threshold)[0]).to(DEVICE)
                num_to_reset = len(underutilized_indices)
                if num_to_reset > 0:
                    print(f"[CODEBOOK MGMT] Resetting {num_to_reset} underutilized codes.")
                    reset_data = next(iter(dataloader))
                    if reset_data and reset_data[0] is not None:
                        motion_batch, _ = reset_data
                        x_reset = motion_batch.to(DEVICE)
                        z_e = vq_model.vqvae.encoder(vq_model.vqvae.preprocess(x_reset))
                        z_e_flat = z_e.permute(0, 2, 1).reshape(-1, z_e.shape[1])
                        if z_e_flat.shape[0] >= num_to_reset:
                            indices = torch.randperm(z_e_flat.size(0))[:num_to_reset]
                            vq_model.vqvae.quantizer.codebook.data[underutilized_indices] = z_e_flat[indices]
            vq_model.train()

        if epoch > 0 and epoch % 5 == 0:
            checkpoint_manager.save_checkpoint(vq_model, optimizer, epoch, np.mean(epoch_losses))

    print("\n✅ Training loop finished.")
    return vq_model


# ==============================================================================
# 6) MAIN EXECUTION SCRIPT (No Globals)
# ==============================================================================
def main_colab():
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive mounted successfully.")

    GDRIVE_ROOT = '/content/drive/MyDrive'

    # Define all paths locally within the main function
    STATS_PATH = f'/content/dataset_stats.pt'
    DATA_ROOT = f'{GDRIVE_ROOT}/kaggle_upload/npz_data/batch_1'
    CHECKPOINT_DIR = f'{GDRIVE_ROOT}/Colab_Checkpoints/MotionGPT_VQVAE_Final'

    # The 'global' keyword is no longer needed
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Data Root: {DATA_ROOT}")
    print(f"Stats Path: {STATS_PATH}")
    print(f"Checkpoint Dir: {CHECKPOINT_DIR}")

    smpl_dim = 182
    codebook_size = 512
    code_dim = 512
    vq_model = MotionGPT_VQVAE_Wrapper(
        smpl_dim=smpl_dim, codebook_size=codebook_size, code_dim=code_dim,
        quantizer="ema_reset", width=512, depth=3, down_t=3, stride_t=2,
        dilation_growth_rate=3, activation='relu', norm=None
    ).to(DEVICE)

    motion_dataset = NpzMotionDataset(
        root_dir=DATA_ROOT,
        stats_path=STATS_PATH,
        min_seq_len=64
    )

    # Pass CHECKPOINT_DIR as an argument to the training function
    vq_model = train_vqvae_colab(
        vq_model,
        motion_dataset,
        checkpoint_dir=CHECKPOINT_DIR, # Pass the path here
        num_epochs=1000,
        batch_size=32,
        lr=2e-4
    )

    print("\n" + "="*70 + "\nVQ-VAE TRAINING COMPLETED SUCCESSFULLY!\n" + "="*70)
    final_model_path = os.path.join(CHECKPOINT_DIR, 'final_vqvae_model.pt')
    torch.save({'model_state_dict': vq_model.state_dict()}, final_model_path)
    print(f"Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main_colab()