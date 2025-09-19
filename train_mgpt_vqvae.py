import os
import pickle
import zipfile
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import warnings
import json
import time
from datetime import datetime
import random
import math
import matplotlib.pyplot as plt
import sys

# Add the mGPT directory to the path
sys.path.append('/kaggle/working')

from mGPT.archs.mgpt_vq import VQVae

warnings.filterwarnings("ignore")

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = '/kaggle/working/extracted_files'
CHECKPOINT_DIR = '/kaggle/working/checkpoints_mgpt'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print("Device:", DEVICE)

# ──────────────────────────────────────────────────────────
# Enhanced Dataset with File Tracking and Batching (UNCHANGED)
# ──────────────────────────────────────────────────────────

def load_smplx_from_folder(folder_path):
    all_frame_dicts = []
    for pkl_file in sorted(glob.glob(os.path.join(folder_path, '*.pkl'))):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    all_frame_dicts.extend(data)
                elif isinstance(data, dict):
                    all_frame_dicts.append(data)
        except Exception:
            continue
    if not all_frame_dicts:
        return None

    param_keys = ['shape','body_pose','lhand_pose','rhand_pose','jaw_pose',
                  'expression','root_pose','cam_trans']
    param_dims = [10,63,45,45,3,10,3,3]
    sequences = []
    for frame in all_frame_dicts:
        vec = []
        for key, dim in zip(param_keys, param_dims):
            arr = np.zeros(dim)
            if key in frame and frame[key] is not None:
                v = np.array(frame[key]).flatten()
                arr[:min(len(v), dim)] = v[:dim]
            vec.append(arr)
        sequences.append(np.concatenate(vec))
    return torch.tensor(np.stack(sequences), dtype=torch.float32)

class EnhancedMotionDataset(Dataset):
    def __init__(self, root_dir, processed_files_path, batch_folders=1000):
        self.root_dir = root_dir
        self.processed_files_path = processed_files_path
        self.batch_folders = batch_folders

        print(f"\n[DEBUG] Initializing Dataset.")
        print(f"[DEBUG] Root directory: '{self.root_dir}'")

        if not os.path.exists(self.root_dir):
            print(f"[DEBUG] ERROR: The root directory '{self.root_dir}' does not exist!")
            self.all_folders = []
        else:
            print(f"[DEBUG] Root directory exists.")
            glob_path = os.path.join(root_dir, '*')
            print(f"[DEBUG] Using glob pattern: '{glob_path}'")
            all_paths = glob.glob(glob_path)
            print(f"[DEBUG] Glob found {len(all_paths)} total paths.")
            self.all_folders = [d for d in all_paths if os.path.isdir(d)]
            print(f"[DEBUG] Found {len(self.all_folders)} directories.")

        self.processed = self._load_processed()
        print(f"[DEBUG] Loaded {len(self.processed)} processed folder paths.")

        self.unprocessed = [f for f in self.all_folders if f not in self.processed]
        print(f"[DEBUG] Found {len(self.unprocessed)} unprocessed folders.")

        self._prep_batch()

    def _load_processed(self):
        if os.path.exists(self.processed_files_path):
            with open(self.processed_files_path, 'r') as f:
                return json.load(f)
        return []

    def _save_processed(self):
        with open(self.processed_files_path, 'w') as f:
            json.dump(self.processed, f)

    def _prep_batch(self):
        self.current = self.unprocessed[:self.batch_folders]
        self.samples = self.current.copy()
        print(f"→ Loading {len(self.samples)} folders this batch")

    def mark_batch_as_processed(self):
        self.processed += self.current
        self._save_processed()

    def get_next_batch(self):
        all_folders = [d for d in glob.glob(os.path.join(self.root_dir, '*')) if os.path.isdir(d)]
        self.processed = self._load_processed()
        self.unprocessed = [f for f in all_folders if f not in self.processed]

        if not self.unprocessed:
            print("✅ All data processed")
            return False
        self._prep_batch()
        return True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = load_smplx_from_folder(self.samples[idx])
        if seq is None or seq.shape[0] < 64:
            return None
        return seq

# ──────────────────────────────────────────────────────────
# Checkpoint Management (UNCHANGED)
# ──────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, checkpoint_dir, max_checkpoints=2):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints

    def save_checkpoint(self, model, optimizer, epoch, batch_idx, loss, metadata=None):
        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'mgpt_vqvae_epoch_{epoch:03d}_batch_{batch_idx:04d}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        self.cleanup_old_checkpoints()
        return checkpoint_path

    def cleanup_old_checkpoints(self):
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'mgpt_vqvae_epoch_*.pt'))
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        if len(checkpoints) > self.max_checkpoints:
            for checkpoint in checkpoints[self.max_checkpoints:]:
                os.remove(checkpoint)
                print(f"Removed old checkpoint: {checkpoint}")

    def load_latest_checkpoint(self):
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'mgpt_vqvae_epoch_*.pt'))
        if not checkpoints:
            return None
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"Loading checkpoint: {latest_checkpoint}")
        return torch.load(latest_checkpoint, map_location=DEVICE)

    def get_checkpoint_info(self):
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'mgpt_vqvae_epoch_*.pt'))
        return len(checkpoints), checkpoints

# ──────────────────────────────────────────────────────────
# Enhanced Training Function with MotionGPT VQ-VAE
# ──────────────────────────────────────────────────────────

def train_mgpt_vqvae(vq_model, dataset, epochs_per_batch=20, batch_size=16, lr=1e-4):
    print("\n" + "="*70)
    print("      STARTING MGPT VQ-VAE TRAINING WITH CHECKPOINTING      ")
    print("="*70)

    optimizer = torch.optim.AdamW(vq_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    loss_fn = nn.SmoothL1Loss(reduction='none')
    checkpoint_manager = CheckpointManager(CHECKPOINT_DIR)

    checkpoint = checkpoint_manager.load_latest_checkpoint()
    global_epoch = 1
    if checkpoint:
        vq_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_epoch = checkpoint.get('metadata', {}).get('global_epoch', checkpoint['epoch'])
        print(f"Resumed from GLOBAL epoch {global_epoch}")

    vq_model.to(DEVICE).train()

    # Define loss weights for SMPL parameters
    param_dims = [10, 63, 45, 45, 3, 10, 3, 3]
    param_starts = np.cumsum([0] + param_dims[:-1]).tolist()
    smpl_dim = sum(param_dims)
    loss_weights = torch.ones(smpl_dim, device=DEVICE)
    loss_weights[param_starts[1]:param_starts[5]] = 10.0  # pose parameters
    loss_weights[param_starts[0]:param_starts[1]] = 5.0   # shape parameters
    loss_weights[param_starts[5]:param_starts[6]] = 8.0   # expression parameters

    def log_codebook_analysis(x_recon, loss, perplexity, epoch, batch_idx):
        # Extract encoded indices for analysis
        with torch.no_grad():
            x_in = vq_model.preprocess(x_recon[:1])  # Use reconstructed sample for analysis
            x_encoder = vq_model.encoder(x_in)
            x_flat = vq_model.quantizer.preprocess(x_encoder)
            indices = vq_model.quantizer.quantize(x_flat)

        unique_codes = torch.unique(indices)
        usage_percentage = (len(unique_codes) / vq_model.quantizer.nb_code) * 100

        print(f"[ANALYSIS] Epoch {epoch}, Batch {batch_idx}")
        print(f"Unique codes used: {len(unique_codes)}/{vq_model.quantizer.nb_code} ({usage_percentage:.1f}%)")
        print(f"Perplexity: {perplexity:.2f}")
        return usage_percentage, indices

    def save_reconstruction_sample(x, x_recon, lengths, epoch):
        original_seq = x[0, :lengths[0]].cpu().numpy()
        recon_seq = x_recon[0, :lengths[0]].cpu().numpy()
        filename = os.path.join(CHECKPOINT_DIR, f'mgpt_recon_epoch_{epoch}.npz')
        np.savez(filename, original=original_seq, reconstructed=recon_seq)
        print(f"Saved reconstruction sample to {filename}")
        mse = ((original_seq - recon_seq) ** 2).mean()
        print(f"Reconstruction MSE: {mse:.6f}")
        return mse

    def collate_fn_enhanced(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None
        batch.sort(key=lambda x: x.shape[0], reverse=True)
        max_len = batch[0].shape[0]
        max_len = min(max_len, 256)
        downsampling_factor = 8
        padded_max_len = math.ceil(max_len / downsampling_factor) * downsampling_factor
        padded_batch = torch.zeros(len(batch), padded_max_len, batch[0].shape[1])
        lengths = []
        for i, x in enumerate(batch):
            length = min(x.shape[0], padded_max_len)
            padded_batch[i, :length, :] = x[:length, :]
            lengths.append(length)
        return padded_batch, torch.tensor(lengths)

    while True:
        print(f"\n{'='*50}")
        print(f"Processing file batch with {len(dataset)} files")
        print(f"{'='*50}")

        if len(dataset) == 0:
            if not dataset.get_next_batch():
                print("✅ All data processed! Training complete.")
                break
            continue

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_fn_enhanced, drop_last=True
        )

        for epoch in range(global_epoch, global_epoch + epochs_per_batch):
            epoch_losses, epoch_vq_losses, epoch_rec_losses = [], [], []
            codebook_usage_history = []
            epoch_indices = []

            for batch_idx, batch_data in enumerate(dataloader):
                if batch_data is None:
                    continue

                motion_batch, lengths = batch_data
                x = motion_batch.to(DEVICE)

                # Forward pass through MotionGPT VQ-VAE
                x_recon, vq_loss, perplexity = vq_model(x)

                if batch_idx % 50 == 0:
                    usage_pct, indices = log_codebook_analysis(x_recon, vq_loss, perplexity, epoch, batch_idx)
                    epoch_indices.append(indices.cpu().numpy().flatten())

                # Calculate reconstruction loss with weighted parameters
                rec_loss_unreduced = loss_fn(x_recon, x) * loss_weights.unsqueeze(0).unsqueeze(0)
                mask = torch.zeros_like(x[:, :, 0])
                for i, length in enumerate(lengths):
                    mask[i, :length] = 1.0
                mask = mask.unsqueeze(-1).expand_as(rec_loss_unreduced)
                rec_loss = (rec_loss_unreduced * mask).sum() / mask.sum()

                vq_weight = 1.0
                total_loss = rec_loss + vq_weight * vq_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(vq_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_losses.append(total_loss.item())
                epoch_vq_losses.append(vq_loss.item())
                epoch_rec_losses.append(rec_loss.item())

                if batch_idx % 20 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"[E:{epoch:03d}] B:{batch_idx:03d} | "
                          f"Loss: {total_loss.item():.4f} "
                          f"(Rec: {rec_loss.item():.4f}, VQ: {vq_loss.item():.4f}) | "
                          f"Perplexity: {perplexity:.2f} | "
                          f"LR: {current_lr:.2e}")

            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                avg_vq_loss = np.mean(epoch_vq_losses)
                avg_rec_loss = np.mean(epoch_rec_losses)

                print(f"\n[EPOCH {epoch:03d} SUMMARY]")
                print(f"Avg Loss: {avg_loss:.4f} (Rec: {avg_rec_loss:.4f}, VQ: {avg_vq_loss:.4f})")

                # Create histogram if we collected indices
                if epoch_indices:
                    all_epoch_indices = np.concatenate(epoch_indices)
                    plt.figure(figsize=(12, 6))
                    plt.hist(all_epoch_indices, bins=vq_model.quantizer.nb_code,
                           range=(0, vq_model.quantizer.nb_code-1))
                    plt.title(f'MotionGPT Codebook Usage Distribution - Epoch {epoch}')
                    plt.xlabel('Codebook Index')
                    plt.ylabel('Frequency')
                    hist_path = os.path.join(CHECKPOINT_DIR, f'mgpt_codebook_usage_epoch_{epoch:03d}.png')
                    plt.savefig(hist_path)
                    plt.close()
                    print(f"Saved codebook usage histogram to {hist_path}")

            if epoch > 0 and epoch % 5 == 0:
                vq_model.eval()
                with torch.no_grad():
                    for val_data in dataloader:
                        if val_data is not None:
                            motion_batch, lengths = val_data
                            x = motion_batch.to(DEVICE)
                            x_recon, _, _ = vq_model(x)
                            save_reconstruction_sample(x, x_recon, lengths, epoch)
                            break
                vq_model.train()

            if epoch > 0 and epoch % 10 == 0:
                checkpoint_manager.save_checkpoint(
                    vq_model, optimizer, epoch, -1, np.mean(epoch_losses),
                    metadata={'global_epoch': epoch}
                )

        global_epoch += epochs_per_batch

        dataset.mark_batch_as_processed()

        if not dataset.get_next_batch():
            print("✅ All data processed! Training complete.")
            break

    return vq_model

# ──────────────────────────────────────────────────────────
# Main Training Script
# ──────────────────────────────────────────────────────────

def main():
    print("Starting MotionGPT VQ-VAE Training System")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")

    smpl_dim = 182
    codebook_size = 512
    code_dim = 512

    # Initialize MotionGPT VQ-VAE
    vq_model = VQVae(
        nfeats=smpl_dim,
        quantizer="ema_reset",  # Options: "ema_reset", "orig", "ema", "reset"
        code_num=codebook_size,
        code_dim=code_dim,
        output_emb_width=code_dim,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        norm=None,
        activation="relu"
    ).to(DEVICE)

    total_params = sum(p.numel() for p in vq_model.parameters())
    trainable_params = sum(p.numel() for p in vq_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    motion_dataset = EnhancedMotionDataset(
        root_dir=DATA_ROOT,
        processed_files_path=os.path.join(CHECKPOINT_DIR, 'processed_folders_mgpt.json'),
        batch_folders=800
    )

    vq_model = train_mgpt_vqvae(
        vq_model,
        motion_dataset,
        epochs_per_batch=15,
        batch_size=12,
        lr=2e-4
    )

    print("\n" + "="*70)
    print("MGPT VQ-VAE TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)

    final_model_path = os.path.join(CHECKPOINT_DIR, 'final_mgpt_vqvae_model.pt')
    torch.save({
        'model_state_dict': vq_model.state_dict(),
        'model_config': {
            'nfeats': smpl_dim,
            'code_num': codebook_size,
            'code_dim': code_dim,
            'quantizer': "ema_reset"
        },
        'training_completed': True
    }, final_model_path)
    print(f"Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main()
