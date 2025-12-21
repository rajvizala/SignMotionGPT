"""
Visualization script to convert motion tokens to SMPL-X 3D animation.
Requires VQ-VAE checkpoint, dataset stats, and SMPL-X model files.

Usage:
    # Visualize from LLM output string
    python visualize.py --tokens "<MOT_BEGIN><motion_177><motion_135>...<MOT_END>"
    
    # Visualize from saved file
    python visualize.py --input motion_output.txt
    
    # Generate and visualize in one go
    python visualize.py --prompt "walking" --stage 3
    
    # Custom paths
    python visualize.py --tokens "..." --vqvae-ckpt /path/to/vqvae.pt --smplx-dir /path/to/smplx
"""
import os
import sys
import re
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import WORK_DIR, DATA_DIR

# Try importing visualization dependencies
try:
    import plotly.graph_objects as go
except ImportError:
    print("Installing plotly...")
    os.system("pip install -q plotly")
    import plotly.graph_objects as go

try:
    import smplx
except ImportError:
    print("Installing smplx...")
    os.system("pip install -q smplx==0.1.28")
    import smplx

# =====================================================================
# Configuration - can be overridden via command-line or environment
# =====================================================================
# VQ-VAE checkpoint path (trained motion encoder/decoder)
VQVAE_CHECKPOINT = os.environ.get(
    "VQVAE_CHECKPOINT",
    os.path.join(DATA_DIR, "vqvae_model.pt")
)

# Dataset normalization stats (mean/std used during VQ-VAE training)
STATS_PATH = os.environ.get(
    "VQVAE_STATS_PATH",
    os.path.join(DATA_DIR, "vqvae_stats.pt")
)

# SMPL-X model directory (contains SMPLX_NEUTRAL.npz, etc.)
SMPLX_MODEL_DIR = os.environ.get(
    "SMPLX_MODEL_DIR",
    os.path.join(DATA_DIR, "smplx_models")
)

# Output directory for HTML animations
OUTPUT_DIR = os.environ.get("VIS_OUTPUT_DIR", WORK_DIR)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VQ-VAE architecture params (must match training config)
SMPL_DIM = 182
CODEBOOK_SIZE = 512
CODE_DIM = 512
VQ_ARGS = dict(
    width=512,
    depth=3,
    down_t=2,
    stride_t=2,
    dilation_growth_rate=3,
    activation='relu',
    norm=None,
    quantizer="ema_reset"
)

# SMPL-X parameter layout (must match VQ-VAE training)
PARAM_DIMS = [10, 63, 45, 45, 3, 10, 3, 3]
PARAM_NAMES = ["betas", "body_pose", "left_hand_pose", "right_hand_pose",
               "trans", "expression", "jaw_pose", "eye_pose"]

# =====================================================================
# Import VQ-VAE architecture
# =====================================================================
try:
    # Add SignMotionGPT to path if not already
    sign_mgpt_dir = os.path.join(os.path.dirname(__file__))
    if sign_mgpt_dir not in sys.path:
        sys.path.insert(0, sign_mgpt_dir)
    
    from mGPT.archs.mgpt_vq import VQVae
except ImportError as e:
    print(f"❌ Could not import VQVae: {e}")
    print("Make sure mGPT/archs/mgpt_vq.py exists in the project.")
    sys.exit(1)


# =====================================================================
# VQ-VAE Wrapper
# =====================================================================
class MotionGPT_VQVAE_Wrapper(nn.Module):
    """Wrapper matching the VQ-VAE training setup"""
    def __init__(self, smpl_dim=SMPL_DIM, codebook_size=CODEBOOK_SIZE, 
                 code_dim=CODE_DIM, **kwargs):
        super().__init__()
        self.vqvae = VQVae(
            nfeats=smpl_dim,
            code_num=codebook_size,
            code_dim=code_dim,
            output_emb_width=code_dim,
            **kwargs
        )


# =====================================================================
# Token Parsing
# =====================================================================
def parse_motion_tokens(token_str):
    """
    Parse motion tokens from LLM output string.
    Accepts:
      - "<MOT_BEGIN><motion_177><motion_135>...<MOT_END>"
      - "177 135 152 200 46..."
      - List/array of ints
    
    Returns:
        List of token integers
    """
    if isinstance(token_str, (list, tuple, np.ndarray)):
        return [int(x) for x in token_str]
    
    if not isinstance(token_str, str):
        raise ValueError("Tokens must be string or list-like")
    
    # Try extracting <motion_ID> or <MID> tokens
    matches = re.findall(r'<motion_(\d+)>|<M(\d+)>', token_str)
    if matches:
        ids = []
        for m_old, m_new in matches:
            if m_old:
                ids.append(int(m_old))
            else:
                ids.append(int(m_new))
        return ids
    
    # Try space-separated numbers
    token_str = token_str.strip()
    if token_str:
        try:
            return [int(x) for x in token_str.split()]
        except ValueError:
            pass
    
    raise ValueError(f"Could not parse motion tokens from: {token_str[:100]}...")


# =====================================================================
# Model Loading
# =====================================================================
def load_vqvae(checkpoint_path, device=DEVICE, vq_args=VQ_ARGS):
    """Load trained VQ-VAE model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"VQ-VAE checkpoint not found: {checkpoint_path}\n"
            f"Please download it and set VQVAE_CHECKPOINT environment variable "
            f"or use --vqvae-ckpt argument."
        )
    
    print(f"Loading VQ-VAE from: {checkpoint_path}")
    model = MotionGPT_VQVAE_Wrapper(
        smpl_dim=SMPL_DIM,
        codebook_size=CODEBOOK_SIZE,
        code_dim=CODE_DIM,
        **vq_args
    ).to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"✅ VQ-VAE loaded (codebook size: {CODEBOOK_SIZE})")
    return model


def load_stats(stats_path):
    """Load normalization statistics (mean/std) used during VQ-VAE training"""
    if not stats_path or not os.path.exists(stats_path):
        print(f"⚠️  Stats file not found: {stats_path}")
        print("   Will skip denormalization (may affect quality)")
        return None, None
    
    print(f"Loading stats from: {stats_path}")
    st = torch.load(stats_path, map_location='cpu', weights_only=False)
    mean = st.get('mean', 0)
    std = st.get('std', 1)
    
    # Convert to numpy
    if torch.is_tensor(mean):
        mean = mean.cpu().numpy()
    if torch.is_tensor(std):
        std = std.cpu().numpy()
    
    print(f"✅ Stats loaded (mean shape: {np.array(mean).shape})")
    return mean, std


def load_smplx_model(model_dir, device=DEVICE):
    """Load SMPL-X body model"""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"SMPL-X model directory not found: {model_dir}\n"
            f"Please download SMPL-X models and set SMPLX_MODEL_DIR environment variable "
            f"or use --smplx-dir argument."
        )
    
    print(f"Loading SMPL-X from: {model_dir}")
    model = smplx.SMPLX(
        model_path=model_dir,
        model_type='smplx',
        gender='neutral',
        use_pca=False,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_expression=True,
        create_jaw_pose=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_transl=True
    ).to(device)
    
    print(f"✅ SMPL-X loaded")
    return model


# =====================================================================
# Token Decoding
# =====================================================================
def decode_tokens_to_params(tokens, vqvae_model, mean=None, std=None, device=DEVICE):
    """
    Decode motion tokens to SMPL-X parameters.
    
    Args:
        tokens: List of motion token IDs
        vqvae_model: Trained VQ-VAE model
        mean: Optional normalization mean
        std: Optional normalization std
        device: Device to run on
    
    Returns:
        numpy array of shape (T, SMPL_DIM) with SMPL-X parameters
    """
    if not tokens:
        return np.zeros((0, SMPL_DIM), dtype=np.float32)
    
    # Prepare token indices
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T_q)
    T_q = idx.shape[1]
    
    quantizer = vqvae_model.vqvae.quantizer
    
    # Get code dimension
    if hasattr(quantizer, "codebook"):
        codebook = quantizer.codebook.to(device)
        code_dim = codebook.shape[1]
    else:
        code_dim = CODE_DIM
    
    # Dequantize tokens
    x_quantized = None
    if hasattr(quantizer, "dequantize"):
        try:
            with torch.no_grad():
                dq = quantizer.dequantize(idx)
            if dq is not None:
                dq = dq.contiguous()
                # Ensure shape is (N, code_dim, T_q)
                if dq.ndim == 3 and dq.shape[1] == code_dim:
                    x_quantized = dq
                elif dq.ndim == 3 and dq.shape[1] == T_q:
                    x_quantized = dq.permute(0, 2, 1).contiguous()
                else:
                    x_quantized = None
        except Exception:
            x_quantized = None
    
    # Fallback: manual codebook lookup
    if x_quantized is None:
        if not hasattr(quantizer, "codebook"):
            raise RuntimeError("No dequantize method and no codebook available")
        with torch.no_grad():
            emb = codebook[idx]  # (1, T_q, code_dim)
            x_quantized = emb.permute(0, 2, 1).contiguous()  # (1, code_dim, T_q)
    
    # Decode through VQ-VAE decoder
    with torch.no_grad():
        x_dec = vqvae_model.vqvae.decoder(x_quantized)
        smpl_out = vqvae_model.vqvae.postprocess(x_dec)  # (1, T_out, SMPL_DIM)
        params_np = smpl_out.squeeze(0).cpu().numpy()  # (T_out, SMPL_DIM)
    
    # Denormalize if stats provided
    if (mean is not None) and (std is not None):
        mean_arr = np.array(mean).reshape(1, -1)
        std_arr = np.array(std).reshape(1, -1)
        params_np = (params_np * std_arr) + mean_arr
    
    return params_np


# =====================================================================
# SMPL-X Parameter to Vertices
# =====================================================================
def params_to_vertices(params_seq, smplx_model, batch_size=32):
    """
    Convert SMPL-X parameters to 3D vertices.
    """
    # Compute parameter slicing indices
    starts = np.cumsum([0] + PARAM_DIMS[:-1])
    ends = starts + np.array(PARAM_DIMS)
    
    T = params_seq.shape[0]
    all_verts = []
    
    # Infer number of body joints
    num_body_joints = getattr(smplx_model, "NUM_BODY_JOINTS", 21)
    
    with torch.no_grad():
        for s in range(0, T, batch_size):
            batch = params_seq[s:s+batch_size] # (B, SMPL_DIM)
            B = batch.shape[0]
            
            # Extract parameters
            np_parts = {}
            for name, st, ed in zip(PARAM_NAMES, starts, ends):
                np_parts[name] = batch[:, st:ed].astype(np.float32)
            
            # Convert to tensors
            tensor_parts = {
                name: torch.from_numpy(arr).to(DEVICE)
                for name, arr in np_parts.items()
            }
            
            # Handle body pose (may or may not include global orient)
            body_t = tensor_parts['body_pose']
            L_body = body_t.shape[1]
            expected_no_go = num_body_joints * 3
            expected_with_go = (num_body_joints + 1) * 3
            
            if L_body == expected_with_go:
                global_orient = body_t[:, :3].contiguous()
                body_pose_only = body_t[:, 3:].contiguous()
            elif L_body == expected_no_go:
                global_orient = torch.zeros((B, 3), dtype=torch.float32, device=DEVICE)
                body_pose_only = body_t
            else:
                # Best-effort fallback
                if L_body > expected_no_go:
                    global_orient = body_t[:, :3].contiguous()
                    body_pose_only = body_t[:, 3:].contiguous()
                else:
                    pad_len = max(0, expected_no_go - L_body)
                    body_pose_only = F.pad(body_t, (0, pad_len))
                    global_orient = torch.zeros((B, 3), dtype=torch.float32, device=DEVICE)
            
            # Call SMPL-X
            out = smplx_model(
                betas=tensor_parts['betas'],
                global_orient=global_orient,
                body_pose=body_pose_only,
                left_hand_pose=tensor_parts['left_hand_pose'],
                right_hand_pose=tensor_parts['right_hand_pose'],
                expression=tensor_parts['expression'],
                jaw_pose=tensor_parts['jaw_pose'],
                leye_pose=tensor_parts['eye_pose'],
                reye_pose=tensor_parts['eye_pose'],
                transl=tensor_parts['trans'],
                return_verts=True
            )
            
            verts = out.vertices.detach().cpu().numpy() # (B, V, 3)
            all_verts.append(verts)
    
    verts_all = np.concatenate(all_verts, axis=0) # (T, V, 3)
    faces = smplx_model.faces.astype(np.int32)
    
    return verts_all, faces


# =====================================================================
# Visualization
# =====================================================================
def animate_motion(verts, faces, title="Generated Motion", output_path=None, fps=20):
    """
    Create interactive 3D animation using Plotly.
    Style: Matte white, semi-transparent mesh with improved lighting.
    """
    T, V, _ = verts.shape
    i, j, k = faces.T.tolist()
    
    # Initial mesh setup
    # We use a light gray/white color, 0.5 opacity.
    # Crucially, we add 'lighting' parameters to create highlights on fingers.
    mesh = go.Mesh3d(
        x=verts[0, :, 0],
        y=verts[0, :, 1],
        z=verts[0, :, 2],
        i=i, j=j, k=k,
        name='Skin',
        color='#eeeeee',      # Matte white color
        opacity=0.5,          # Transparency (adjust between 0.1 and 1.0)
        flatshading=False,    # Smooth shading looks better for curved fingers
        lighting=dict(        # Add specular highlights to define shape
            ambient=0.5,
            diffuse=0.5,
            roughness=0.1,
            specular=0.4
        ),
        lightposition=dict(x=0, y=4, z=2) # Light from top-front
    )
    
    # Create frames for animation
    frames = [
        go.Frame(
            data=[go.Mesh3d(
                x=verts[t, :, 0],
                y=verts[t, :, 1],
                z=verts[t, :, 2],
                i=i, j=j, k=k
                # Note: color/opacity/lighting are inherited from initial mesh
            )],
            name=str(t)
        )
        for t in range(T)
    ]
    
    # Create figure
    fig = go.Figure(data=[mesh], frames=frames)
    
    fig.update_layout(
        title_text=title,
        scene=dict(
            aspectmode='data',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            # Zoomed in slightly closer to see hands better
            camera=dict(eye=dict(x=0, y=-1.5, z=0.5))
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": 1000//fps, "redraw": True},
                        "fromcurrent": True
                    }]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], {
                        "frame": {"duration": 0, "redraw": False}
                    }]
                )
            ]
        )]
    )
    
    # Save HTML
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Animation saved to: {output_path}")
    
    return fig# =====================================================================
# Main Visualization Pipeline
# =====================================================================
def visualize(
    tokens,
    vqvae_ckpt=VQVAE_CHECKPOINT,
    stats_path=STATS_PATH,
    smplx_dir=SMPLX_MODEL_DIR,
    output_html=None,
    title="Generated Motion",
    fps=20
):
    """
    Complete visualization pipeline: tokens -> vertices -> animation.
    
    Args:
        tokens: Motion tokens (string or list of ints)
        vqvae_ckpt: Path to VQ-VAE checkpoint
        stats_path: Path to normalization stats
        smplx_dir: Path to SMPL-X model directory
        output_html: Path to save HTML animation
        title: Animation title
        fps: Frames per second
    
    Returns:
        Plotly figure object
    """
    print("="*60)
    print("Motion Visualization Pipeline")
    print("="*60)
    
    # Parse tokens
    print("\n[1/5] Parsing tokens...")
    token_list = parse_motion_tokens(tokens)
    print(f"   Parsed {len(token_list)} tokens")
    if not token_list:
        print("❌ No tokens to visualize")
        return None
    
    # Load models
    print("\n[2/5] Loading VQ-VAE...")
    vq_model = load_vqvae(vqvae_ckpt, device=DEVICE)
    
    print("\n[3/5] Loading normalization stats...")
    mean, std = load_stats(stats_path)
    
    print("\n[4/5] Loading SMPL-X model...")
    smplx_model = load_smplx_model(smplx_dir, device=DEVICE)
    
    # Decode tokens
    print("\n[5/5] Decoding and rendering...")
    print("   Decoding tokens to SMPL-X parameters...")
    params = decode_tokens_to_params(token_list, vq_model, mean, std, device=DEVICE)
    print(f"   Decoded params shape: {params.shape}")
    
    if params.shape[0] == 0:
        print("❌ No frames produced from decoder")
        return None
    
    # Convert to vertices
    print("   Converting parameters to vertices...")
    verts, faces = params_to_vertices(params, smplx_model, batch_size=32)
    
    # verts, joints, faces = params_to_vertices(params, smplx_model, batch_size=32)
    # print(f" Vertices: {verts.shape}, Joints: {joints.shape}")
    # print(f"   Vertices shape: {verts.shape}, Faces: {faces.shape}")
    
    # Create animation
    print("   Creating animation...")
    if output_html is None:
        output_html = os.path.join(OUTPUT_DIR, "motion_animation.html")
    fig = animate_motion(verts, faces, title=title, output_path=output_html, fps=fps)
    # fig = animate_motion(verts, joints, faces, title=title, output_path=output_html, fps=fps)    
    print("\n" + "="*60)
    print("✅ Visualization complete!")
    print("="*60)
    
    return fig


# =====================================================================
# CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visualize motion tokens as 3D SMPL-X animation"
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--tokens",
        type=str,
        help="Motion tokens string (e.g., '<MOT_BEGIN><motion_177>...<MOT_END>' or '177 135 152...')"
    )
    input_group.add_argument(
        "--input",
        type=str,
        help="Path to file containing motion tokens"
    )
    input_group.add_argument(
        "--prompt",
        type=str,
        help="Generate tokens from text prompt first (requires --stage)"
    )
    
    # Generation options (if using --prompt)
    parser.add_argument(
        "--stage",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Stage model to use for generation (default: 3)"
    )
    
    # Model paths
    parser.add_argument(
        "--vqvae-ckpt",
        type=str,
        default=VQVAE_CHECKPOINT,
        help=f"Path to VQ-VAE checkpoint (default: {VQVAE_CHECKPOINT})"
    )
    parser.add_argument(
        "--stats",
        type=str,
        default=STATS_PATH,
        help=f"Path to normalization stats (default: {STATS_PATH})"
    )
    parser.add_argument(
        "--smplx-dir",
        type=str,
        default=SMPLX_MODEL_DIR,
        help=f"Path to SMPL-X model directory (default: {SMPLX_MODEL_DIR})"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save HTML animation (default: motion_animation.html)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Generated Motion",
        help="Animation title"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for animation (default: 20)"
    )
    
    args = parser.parse_args()
    
    # Get tokens
    if args.prompt:
        # Generate tokens first using inference.py
        print("Generating motion tokens from prompt...")
        from inference import inference
        tokens = inference(
            prompt=args.prompt,
            stage=args.stage,
            output_file=None,
            per_prompt_vocab=True
        )
    elif args.input:
        # Read from file
        with open(args.input, 'r') as f:
            tokens = f.read().strip()
    else:
        # Direct token string
        tokens = args.tokens
    
    # Visualize
    visualize(
        tokens=tokens,
        vqvae_ckpt=args.vqvae_ckpt,
        stats_path=args.stats,
        smplx_dir=args.smplx_dir,
        output_html=args.output,
        title=args.title,
        fps=args.fps
    )


if __name__ == "__main__":
    main()