"""
Visualization script to convert motion tokens to SMPL-X 3D animation.
Requires VQ-VAE checkpoint, dataset stats, and SMPL-X model files.

Usage:
    # Visualize from LLM output string (interactive HTML)
    python visualize.py --tokens "<MOT_BEGIN><motion_177><motion_135>...<MOT_END>"
    
    # Visualize from saved file
    python visualize.py --input motion_output.txt
    
    # Generate and visualize in one go
    python visualize.py --prompt "walking" --stage 3
    
    # High-quality video rendering (like SOKE paper)
    python visualize.py --tokens "..." --render-mode video --output motion.mp4
    
    # Different render styles
    python visualize.py --tokens "..." --render-mode video --style silhouette
    python visualize.py --tokens "..." --render-mode video --style hands-highlight
    
    # Custom paths
    python visualize.py --tokens "..." --vqvae-ckpt /path/to/vqvae.pt --smplx-dir /path/to/smplx
"""
# IMPORTANT: Set OpenGL platform BEFORE any OpenGL imports (for headless rendering)
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"  # Use EGL for headless servers (Colab, etc.)

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

# Optional: PyRender for high-quality rendering (like SOKE paper)
PYRENDER_AVAILABLE = False
try:
    import trimesh
    import pyrender
    from PIL import Image
    PYRENDER_AVAILABLE = True
except ImportError:
    pass  # Will be installed on demand if video rendering requested

# =====================================================================
# Configuration - can be overridden via command-line or environment
# =====================================================================
# VQ-VAE checkpoint path (trained motion encoder/decoder)
VQVAE_CHECKPOINT = "/content/SignMotionGPT/vqvae_finetuned_epoch_1180.pt"
# Dataset normalization stats (mean/std used during VQ-VAE training)
STATS_PATH = "/content/combined_stats.pt"

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

# =============================================================================
# SMPL-X Parameter Layouts - CRITICAL: Must match VQ-VAE training data format!
# =============================================================================

# WORD-LEVEL parameter layout (original word-level VQ-VAE training)
# Order: betas, body_pose, left_hand, right_hand, trans, expression, jaw, eye
WORD_PARAM_DIMS = [10, 63, 45, 45, 3, 10, 3, 3]
WORD_PARAM_NAMES = ["betas", "body_pose", "left_hand_pose", "right_hand_pose",
                    "trans", "expression", "jaw_pose", "eye_pose"]

# SENTENCE-LEVEL parameter layout (How2Sign / hybrid training)
# Order: shape, body_pose, lhand, rhand, jaw, expression, root_pose, cam_trans
SENTENCE_PARAM_DIMS = [10, 63, 45, 45, 3, 10, 3, 3]
SENTENCE_PARAM_NAMES = ["shape", "body_pose", "lhand_pose", "rhand_pose",
                        "jaw_pose", "expression", "root_pose", "cam_trans"]

# Default to sentence-level (hybrid pipeline)
PARAM_DIMS = SENTENCE_PARAM_DIMS
PARAM_NAMES = SENTENCE_PARAM_NAMES

# Backward compatibility aliases
DEFAULT_DATA_LEVEL = "sentence"  # Changed default to sentence for hybrid pipeline

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
def params_to_vertices(params_seq, smplx_model, batch_size=32, data_level="sentence", lock_trans=False):
    """
    Convert SMPL-X parameters to 3D vertices.
    
    Args:
        params_seq: (T, 182) array of SMPL-X parameters
        smplx_model: SMPL-X model
        batch_size: Batch size for processing
        data_level: "word" for word-level data, "sentence" for sentence-level (How2Sign)
                   This affects how parameters are mapped to SMPL-X.
        lock_trans: If True, zero out global translation (stabilizes visualization)
    
    Returns:
        verts_all: (T, V, 3) vertex positions
        faces: (F, 3) face indices
    """
    # Select parameter configuration based on data level
    if data_level == "word":
        param_dims = WORD_PARAM_DIMS
        param_names = WORD_PARAM_NAMES
    else:
        param_dims = SENTENCE_PARAM_DIMS
        param_names = SENTENCE_PARAM_NAMES
    
    # Compute parameter slicing indices
    starts = np.cumsum([0] + param_dims[:-1])
    ends = starts + np.array(param_dims)
    
    T = params_seq.shape[0]
    all_verts = []
    
    # Infer number of body joints
    num_body_joints = getattr(smplx_model, "NUM_BODY_JOINTS", 21)
    
    with torch.no_grad():
        for s in range(0, T, batch_size):
            batch = params_seq[s:s+batch_size]  # (B, SMPL_DIM)
            B = batch.shape[0]
            
            # Extract parameters
            np_parts = {}
            for name, st, ed in zip(param_names, starts, ends):
                np_parts[name] = batch[:, st:ed].astype(np.float32)
            
            # Convert to tensors
            tensor_parts = {
                name: torch.from_numpy(arr).to(DEVICE)
                for name, arr in np_parts.items()
            }
            
            # Map parameters based on data level
            if data_level == "word":
                # Word-level format:
                # - betas -> betas
                # - body_pose may include global_orient
                # - trans -> transl
                # - eye_pose -> leye/reye
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
                    if L_body > expected_no_go:
                        global_orient = body_t[:, :3].contiguous()
                        body_pose_only = body_t[:, 3:].contiguous()
                    else:
                        pad_len = max(0, expected_no_go - L_body)
                        body_pose_only = F.pad(body_t, (0, pad_len))
                        global_orient = torch.zeros((B, 3), dtype=torch.float32, device=DEVICE)
                
                # Handle translation - use zeros if lock_trans is True
                if lock_trans:
                    transl = torch.zeros((B, 3), dtype=torch.float32, device=DEVICE)
                else:
                    transl = tensor_parts['trans']
                
                # Call SMPL-X with word-level mapping
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
                    transl=transl,
                    return_verts=True
                )
            else:
                # Sentence-level format (How2Sign):
                # - shape -> betas
                # - body_pose (63 dims, no global_orient)
                # - lhand_pose, rhand_pose
                # - jaw_pose at position 4
                # - root_pose -> global_orient
                # - cam_trans -> transl
                body_t = tensor_parts['body_pose']
                L_body = body_t.shape[1]
                expected_body = num_body_joints * 3  # 63
                
                if L_body >= expected_body:
                    body_pose_only = body_t[:, :expected_body].contiguous()
                else:
                    pad_len = expected_body - L_body
                    body_pose_only = F.pad(body_t, (0, pad_len))
                
                # Handle translation - use zeros if lock_trans is True
                if lock_trans:
                    transl = torch.zeros((B, 3), dtype=torch.float32, device=DEVICE)
                else:
                    transl = tensor_parts['cam_trans']
                
                # Call SMPL-X with sentence-level mapping
                out = smplx_model(
                    betas=tensor_parts['shape'],
                    global_orient=tensor_parts['root_pose'],
                    body_pose=body_pose_only,
                    left_hand_pose=tensor_parts['lhand_pose'],
                    right_hand_pose=tensor_parts['rhand_pose'],
                    expression=tensor_parts['expression'],
                    jaw_pose=tensor_parts['jaw_pose'],
                    leye_pose=torch.zeros((B, 3), dtype=torch.float32, device=DEVICE),
                    reye_pose=torch.zeros((B, 3), dtype=torch.float32, device=DEVICE),
                    transl=transl,
                    return_verts=True
                )
            
            verts = out.vertices.detach().cpu().numpy()  # (B, V, 3)
            all_verts.append(verts)
    
    verts_all = np.concatenate(all_verts, axis=0)  # (T, V, 3)
    faces = smplx_model.faces.astype(np.int32)
    
    return verts_all, faces


# =====================================================================
# Visualization
# =====================================================================
def animate_motion(verts, faces, title="Generated Motion", output_path=None, fps=20):
    """
    Create interactive 3D animation using Plotly.
    Improved for sign language: frontal view, edge highlighting, better depth perception.
    """
    T, V, _ = verts.shape
    i, j, k = faces.T.tolist()
    
    # Compute vertex colors based on depth (helps see hands in front of body)
    # Vertices closer to camera (more negative y in frontal view) get slightly brighter
    def compute_depth_colors(frame_verts):
        """Create subtle depth-based coloring to enhance 3D perception"""
        y_vals = frame_verts[:, 1]  # depth axis in frontal view
        y_min, y_max = y_vals.min(), y_vals.max()
        if y_max - y_min > 1e-6:
            depth_norm = (y_vals - y_min) / (y_max - y_min)  # 0 = closest, 1 = farthest
        else:
            depth_norm = np.zeros_like(y_vals)
        
        # Closer vertices get warmer tone (helps see hands), farther get cooler
        # Base color: warm cream for close, cooler gray for far
        r = 0.95 - 0.15 * depth_norm
        g = 0.90 - 0.20 * depth_norm
        b = 0.85 - 0.10 * depth_norm
        
        # Convert to hex colors
        colors = [f'rgb({int(r[i]*255)},{int(g[i]*255)},{int(b[i]*255)})' 
                  for i in range(len(r))]
        return colors
    
    # Initial mesh with improved settings for sign language visualization
    initial_colors = compute_depth_colors(verts[0])
    
    mesh = go.Mesh3d(
        x=verts[0, :, 0],
        y=verts[0, :, 1],
        z=verts[0, :, 2],
        i=i, j=j, k=k,
        name='Body',
        vertexcolor=initial_colors,  # Depth-based vertex colors
        opacity=1.0,                  # Opaque for clear visibility
        flatshading=False,            # Smooth shading
        lighting=dict(
            ambient=0.6,              # Higher ambient for overall visibility
            diffuse=0.6,              # Good diffuse for shape definition
            roughness=0.3,            # Slightly rough for matte look
            specular=0.3,             # Subtle specular for finger highlights
            fresnel=0.2               # Edge highlighting effect
        ),
        lightposition=dict(x=0, y=-5, z=3)  # Light from front-top
    )
    
    # Create frames with depth-based coloring
    frames = []
    for t in range(T):
        frame_colors = compute_depth_colors(verts[t])
        frames.append(go.Frame(
            data=[go.Mesh3d(
                x=verts[t, :, 0],
                y=verts[t, :, 1],
                z=verts[t, :, 2],
                i=i, j=j, k=k,
                vertexcolor=frame_colors
            )],
            name=str(t)
        ))
    
    # Create figure
    fig = go.Figure(data=[mesh], frames=frames)
    
    # Frontal camera for sign language (looking at the signer from the front)
    fig.update_layout(
        title_text=title,
        scene=dict(
            aspectmode='data',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgb(240, 240, 245)',  # Light gray background
            # Frontal view - sign language is viewed from front
            camera=dict(
                eye=dict(x=0, y=-2.0, z=0.3),    # Front view, slightly below eye level
                up=dict(x=0, y=0, z=1),          # Z is up
                center=dict(x=0, y=0, z=0)
            )
        ),
        paper_bgcolor='rgb(240, 240, 245)',
        updatemenus=[dict(
            type="buttons",
            showactive=True,
            y=1.0,
            x=0.5,
            xanchor="center",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": 1000//fps, "redraw": True},
                        "fromcurrent": True,
                        "mode": "immediate"
                    }]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate"
                    }]
                )
            ]
        )],
        sliders=[dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(
                font=dict(size=12),
                prefix="Frame: ",
                visible=True,
                xanchor="center"
            ),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.05,
            y=0,
            steps=[dict(
                args=[[str(t)], dict(
                    frame=dict(duration=0, redraw=True),
                    mode="immediate"
                )],
                label=str(t),
                method="animate"
            ) for t in range(T)]
        )]
    )
    
    # Save HTML
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"Animation saved to: {output_path}")
    
    return fig


# =====================================================================
# High-Quality Video Rendering (PyRender - like SOKE paper)
# =====================================================================
def ensure_pyrender():
    """Install pyrender dependencies if not available (with EGL support for headless)"""
    global PYRENDER_AVAILABLE, trimesh, pyrender, Image
    if PYRENDER_AVAILABLE:
        return True
    
    print("Installing pyrender dependencies for high-quality rendering...")
    
    # Install EGL support for headless rendering (needed on Colab/servers)
    # libegl1-mesa-dev provides EGL support
    if os.path.exists("/etc/debian_version"):  # Debian/Ubuntu/Colab
        os.system("apt-get update -qq && apt-get install -qq -y libegl1-mesa-dev libgles2-mesa-dev > /dev/null 2>&1")
    
    # Install Python packages
    os.system("pip install -q trimesh pyrender PyOpenGL PyOpenGL_accelerate Pillow opencv-python imageio[ffmpeg]")
    
    try:
        import trimesh
        import pyrender
        from PIL import Image
        PYRENDER_AVAILABLE = True
        return True
    except ImportError as e:
        print(f"Could not install pyrender: {e}")
        print("Falling back to Plotly rendering.")
        return False


# SMPL-X vertex indices for hands (approximate - for highlighting)
# These are rough indices for left/right hand vertices
HAND_VERTEX_INDICES = {
    'left_hand': list(range(5443, 5672)),   # Left hand vertices
    'right_hand': list(range(8017, 8246)),  # Right hand vertices
}


def render_mesh_pyrender(
    verts, 
    faces, 
    img_size=(1080, 1080),
    focal_length=2000,
    camera_distance=3.5,
    style='default',
    bg_color=(0.95, 0.95, 0.97, 1.0),
    fixed_center=None,
    apply_rotation=True
):
    """
    Render a single mesh frame using PyRender (high quality).
    
    Args:
        verts: (V, 3) vertex positions
        faces: (F, 3) face indices
        img_size: (width, height) output image size
        focal_length: Camera focal length (lower = wider view, higher = more zoom)
                      Default 2000 shows full body well
        camera_distance: Distance from camera to subject (higher = more zoomed out)
         style: 'default', 'silhouette', 'hands-highlight', 'wireframe-overlay'
         bg_color: Background color RGBA
        fixed_center: Optional fixed camera target (reduces jitter). Should be computed from rotated vertices.
        apply_rotation: If True, apply 180-degree rotation around X-axis. Set False if vertices already rotated.
    
    Returns:
        RGB image as numpy array (H, W, 3)
    """
    if not PYRENDER_AVAILABLE:
        raise RuntimeError("PyRender not available. Call ensure_pyrender() first.")
    
    width, height = img_size
    
    # Create scene
    scene = pyrender.Scene(bg_color=bg_color, ambient_light=[0.4, 0.4, 0.4])
    
    # IMPORTANT: Rotate mesh 180 degrees around X-axis (like SOKE paper)
    # This fixes the coordinate system so we view from the front
    # Only apply if not already rotated by caller
    if apply_rotation:
        rot_matrix = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        verts_rotated = np.dot(verts, rot_matrix[:3, :3].T)
    else:
        verts_rotated = verts
    
    # Create mesh with style-dependent material
    # Use rotated vertices for all mesh creation
    if style == 'silhouette':
        # Dark silhouette style (good for publication figures)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.9,
            alphaMode='OPAQUE',
            baseColorFactor=(0.15, 0.15, 0.18, 1.0)
        )
        mesh = trimesh.Trimesh(vertices=verts_rotated, faces=faces)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
        scene.add(mesh)
        
    elif style == 'hands-highlight':
        # Body in neutral color, hands in highlighted color
        # This makes hand motion clearly visible
        
        # Create vertex colors: neutral gray for body, warm color for hands
        vertex_colors = np.ones((verts_rotated.shape[0], 4)) * 0.85  # Light gray
        vertex_colors[:, 3] = 1.0
        
        # Highlight hands with warm tone
        for hand_name, indices in HAND_VERTEX_INDICES.items():
            valid_indices = [i for i in indices if i < verts_rotated.shape[0]]
            vertex_colors[valid_indices, 0] = 0.95  # R
            vertex_colors[valid_indices, 1] = 0.75  # G
            vertex_colors[valid_indices, 2] = 0.65  # B
        
        mesh = trimesh.Trimesh(vertices=verts_rotated, faces=faces, vertex_colors=vertex_colors)
        mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        scene.add(mesh)
        
    elif style == 'wireframe-overlay':
        # Solid mesh with wireframe overlay (shows structure clearly)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.5,
            alphaMode='OPAQUE',
            baseColorFactor=(0.92, 0.90, 0.88, 1.0)
        )
        mesh = trimesh.Trimesh(vertices=verts_rotated, faces=faces)
        mesh_render = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
        scene.add(mesh_render)
        
    else:
        # Default: Clean cream/white material (like SOKE paper)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.4,
            alphaMode='OPAQUE',
            baseColorFactor=(0.98, 0.96, 0.92, 1.0)  # Warm cream white
        )
        mesh = trimesh.Trimesh(vertices=verts_rotated, faces=faces)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
        scene.add(mesh)
    
    # Compute camera position for frontal view (using rotated vertices)
    # If fixed_center is provided, use it directly (it's already computed from rotated vertices)
    if fixed_center is None:
        center = verts_rotated.mean(axis=0)
    else:
        center = np.asarray(fixed_center)
    
    # Compute bounding box to auto-adjust framing
    bbox_min = verts_rotated.min(axis=0)
    bbox_max = verts_rotated.max(axis=0)
    body_height = bbox_max[1] - bbox_min[1]
    
    # Camera setup - frontal view for sign language
    # Lower focal length = wider field of view = see more of the body
    camera = pyrender.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=width/2, cy=height/2,
        znear=0.1, zfar=20.0
    )
    
    # Camera pose: positioned BEHIND the subject (at +Z), looking at -Z (toward face)
    # After 180-degree rotation of vertices (Y,Z flipped), -Y is up, -Z is forward
    # We position camera at +Z, looking at -Z
    camera_pose = np.eye(4)
    camera_pose[0, 3] = center[0]                    # Center X
    camera_pose[1, 3] = center[1]                    # Center Y
    camera_pose[2, 3] = center[2] + camera_distance  # BEHIND (positive Z)
    
    # Camera orientation: Identity = look at -Z, Up is +Y
    # This matches compare_vqvae.py
    camera_pose[:3, :3] = np.eye(3)
    
    scene.add(camera, pose=camera_pose)
    
    # Lighting setup for clear hand visibility
    # Key light (main light from front-top-right)
    key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    key_pose = np.eye(4)
    key_pose[:3, :3] = trimesh.transformations.euler_matrix(
        np.radians(-30), np.radians(-20), 0
    )[:3, :3]
    scene.add(key_light, pose=key_pose)
    
    # Fill light (softer light from front-left)
    fill_light = pyrender.DirectionalLight(color=[0.9, 0.9, 1.0], intensity=1.5)
    fill_pose = np.eye(4)
    fill_pose[:3, :3] = trimesh.transformations.euler_matrix(
        np.radians(-20), np.radians(30), 0
    )[:3, :3]
    scene.add(fill_light, pose=fill_pose)
    
    # Rim light from behind (helps define silhouette and finger edges)
    rim_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    rim_pose = np.eye(4)
    rim_pose[:3, :3] = trimesh.transformations.euler_matrix(
        np.radians(30), np.radians(180), 0
    )[:3, :3]
    scene.add(rim_light, pose=rim_pose)
    
    # Render with error handling for headless environments
    try:
        renderer = pyrender.OffscreenRenderer(width, height)
    except Exception as e:
        error_msg = str(e)
        if "display" in error_msg.lower() or "egl" in error_msg.lower():
            raise RuntimeError(
                f"PyRender failed to initialize renderer: {e}\n\n"
                "This usually means EGL is not available. Try:\n"
                "  1. Run: apt-get install -y libegl1-mesa-dev libgles2-mesa-dev\n"
                "  2. Restart your Python kernel/runtime\n"
                "  3. Make sure PYOPENGL_PLATFORM=egl is set BEFORE importing pyrender"
            )
        raise
    
    color, depth = renderer.render(scene)
    renderer.delete()
    
    return color


def render_video_pyrender(
    verts_seq, 
    faces, 
    output_path,
    fps=15,
    img_size=(1080, 1080),
    style='default',
    bg_color=(0.95, 0.95, 0.97, 1.0),
    show_progress=True,
    slowdown=2,
    zoom=1.0,
    apply_rotation_fix=True,
    trim_end_frames=True,
    stabilize_motion=True
):
    """
    Render full motion sequence to video using PyRender.
    
    Args:
        verts_seq: (T, V, 3) vertex sequence
        faces: (F, 3) face indices  
        output_path: Path to output video file (.mp4)
        fps: Frames per second (default: 15 for smoother sign language viewing)
        img_size: (width, height) output image size
        style: Rendering style - 'default', 'silhouette', 'hands-highlight'
        bg_color: Background color RGBA
        show_progress: Show progress bar
        slowdown: Factor to slow down video (2 = half speed, 3 = third speed, etc.)
                  Each frame will be repeated 'slowdown' times.
        zoom: Zoom level (0.5 = zoomed out, 1.0 = default, 2.0 = zoomed in)
        apply_rotation_fix: Apply 180-degree rotation around X-axis (fixes orientation)
        trim_end_frames: Trim last few frames to avoid end-of-sequence artifacts
        stabilize_motion: Stabilize the mesh center across frames
    
    Returns:
        Path to output video
    """
    if not ensure_pyrender():
        raise RuntimeError("PyRender not available for video rendering")
    
    try:
        import imageio
    except ImportError:
        print("Installing imageio...")
        os.system("pip install -q imageio[ffmpeg]")
        import imageio
    
    # Make a copy to avoid modifying original
    verts_seq = verts_seq.copy()
    
    # FIX 1: Apply 180-degree rotation around X-axis (from compare_vqvae.py)
    # This fixes "upside down" and "backwards" orientation issues
    if apply_rotation_fix:
        # Transformation: (x, y, z) -> (x, -y, -z)
        verts_seq[..., 1:] *= -1
    
    # FIX 2: Trim end frames to avoid snapping/glitching artifacts
    T_total = verts_seq.shape[0]
    if trim_end_frames and T_total > 10:
        # Trim last 8 frames or 15% of video (whichever is smaller)
        trim_amount = min(8, int(T_total * 0.15))
        T = T_total - trim_amount
        print(f"  Trimming last {trim_amount} frames to remove end-of-sequence artifacts.")
    else:
        T = T_total
    
    # Compute fixed camera target from first frame (after rotation)
    # CRITICAL: Do NOT recenter vertices - just use fixed_center for camera positioning
    # This matches the working approach in compare_vqvae.py and hugging_face_app.py
    fixed_center = verts_seq[0].mean(axis=0)

    total_frames = T * slowdown
    duration_sec = total_frames / fps
    
    # Calculate camera distance from zoom level
    # zoom=0.5 -> further away (camera_distance=5.0), zoom=1.0 -> default (3.5), zoom=2.0 -> closer (2.0)
    camera_distance = 3.5 / zoom
    
    print(f"Rendering {T} motion frames at {img_size[0]}x{img_size[1]}...")
    print(f"  Slowdown: {slowdown}x -> {total_frames} video frames")
    print(f"  FPS: {fps} -> Duration: {duration_sec:.1f} seconds")
    print(f"  Zoom: {zoom}x (camera distance: {camera_distance:.1f})")
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Render all frames
    frames = []
    for t in range(T):
        if show_progress and t % 10 == 0:
            print(f"  Rendering frame {t+1}/{T}...")
        
        frame = render_mesh_pyrender(
            verts_seq[t], 
            faces, 
            img_size=img_size,
            camera_distance=camera_distance,
            style=style,
            bg_color=bg_color,
            fixed_center=fixed_center,
            apply_rotation=False  # Already rotated above
        )
        # Repeat frame for slowdown effect
        for _ in range(slowdown):
            frames.append(frame)
    
    # Write video
    print(f"Writing video to {output_path}...")
    
    # Use imageio for video writing
    if output_path.endswith('.gif'):
        imageio.mimsave(output_path, frames, fps=fps)
    else:
        imageio.mimsave(output_path, frames, fps=fps, codec='libx264', quality=8)
    
    print(f"Video saved to: {output_path} ({duration_sec:.1f}s)")
    return output_path


def render_frames_pyrender(
    verts_seq, 
    faces, 
    output_dir,
    img_size=(1080, 1080),
    style='default',
    bg_color=(0.95, 0.95, 0.97, 1.0),
    show_progress=True
):
    """
    Render motion sequence to individual frame images.
    
    Useful for creating custom videos or figure panels.
    """
    if not ensure_pyrender():
        raise RuntimeError("PyRender not available for frame rendering")
    
    T = verts_seq.shape[0]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Rendering {T} frames to {output_dir}...")
    
    frame_paths = []
    for t in range(T):
        if show_progress and t % 10 == 0:
            print(f"  Rendering frame {t+1}/{T}...")
        
        frame = render_mesh_pyrender(
            verts_seq[t], 
            faces, 
            img_size=img_size,
            style=style,
            bg_color=bg_color
        )
        
        frame_path = os.path.join(output_dir, f"frame_{t:05d}.png")
        Image.fromarray(frame).save(frame_path)
        frame_paths.append(frame_path)
    
    print(f"Frames saved to: {output_dir}")
    return frame_paths


# =====================================================================
# Side-by-Side Comparison Video (GT vs Generated)
# =====================================================================
def render_side_by_side_frame(
    verts_left: np.ndarray,
    verts_right: np.ndarray,
    faces: np.ndarray,
    labels: tuple = ("Ground Truth", "Generated"),
    fixed_center: np.ndarray = None,
    camera_distance: float = 3.5,
    focal_length: float = 2000,
    frame_width: int = 540,
    frame_height: int = 720,
    bg_color: tuple = (0.95, 0.95, 0.97, 1.0)
) -> np.ndarray:
    """
    Render two meshes side-by-side for comparison (single frame).
    
    Args:
        verts_left: (V, 3) vertex positions for left mesh (e.g., Ground Truth)
        verts_right: (V, 3) vertex positions for right mesh (e.g., Generated)
        faces: (F, 3) face indices
        labels: Tuple of labels for (left, right) meshes
        fixed_center: Fixed camera target (use first frame center for stability)
        camera_distance: Distance from camera to subject
        focal_length: Camera focal length
        frame_width: Width of each mesh frame
        frame_height: Height of each mesh frame
        bg_color: Background color RGBA
    
    Returns:
        Combined RGB image as numpy array (H, W*2, 3)
    """
    if not PYRENDER_AVAILABLE:
        raise RuntimeError("PyRender not available. Call ensure_pyrender() first.")
    
    from PIL import Image, ImageDraw, ImageFont
    
    frames = []
    verts_list = [verts_left, verts_right]
    
    # Colors for each mesh type
    colors = [
        (0.3, 0.8, 0.4, 1.0),    # Green for Ground Truth
        (0.3, 0.6, 0.9, 1.0),    # Blue for Generated
    ]
    
    for i, verts in enumerate(verts_list):
        # Check for invalid vertices
        if not np.isfinite(verts).all():
            blank = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 128
            frames.append(blank)
            continue
        
        verts_used = verts.copy()
        mesh_center = verts_used.mean(axis=0)
        camera_target = fixed_center if fixed_center is not None else mesh_center
        
        # Create scene
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=[0.4, 0.4, 0.4])
        
        # Material with distinct color
        color = colors[i % len(colors)]
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.4,
            alphaMode='OPAQUE',
            baseColorFactor=color
        )
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=verts_used, faces=faces)
        mesh_render = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
        scene.add(mesh_render)
        
        # Camera setup
        camera = pyrender.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=frame_width / 2, cy=frame_height / 2,
            znear=0.1, zfar=20.0
        )
        
        # Camera pose: positioned BEHIND the subject, looking at -Z
        camera_pose = np.eye(4)
        camera_pose[0, 3] = camera_target[0]
        camera_pose[1, 3] = camera_target[1]
        camera_pose[2, 3] = camera_target[2] + camera_distance
        camera_pose[:3, :3] = np.eye(3)
        
        scene.add(camera, pose=camera_pose)
        
        # Lighting
        key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        key_pose = np.eye(4)
        key_pose[:3, :3] = trimesh.transformations.euler_matrix(np.radians(-30), np.radians(-20), 0)[:3, :3]
        scene.add(key_light, pose=key_pose)
        
        fill_light = pyrender.DirectionalLight(color=[0.9, 0.9, 1.0], intensity=1.5)
        fill_pose = np.eye(4)
        fill_pose[:3, :3] = trimesh.transformations.euler_matrix(np.radians(-20), np.radians(30), 0)[:3, :3]
        scene.add(fill_light, pose=fill_pose)
        
        rim_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        rim_pose = np.eye(4)
        rim_pose[:3, :3] = trimesh.transformations.euler_matrix(np.radians(30), np.radians(180), 0)[:3, :3]
        scene.add(rim_light, pose=rim_pose)
        
        # Render
        renderer = pyrender.OffscreenRenderer(viewport_width=frame_width, viewport_height=frame_height, point_size=1.0)
        color_img, _ = renderer.render(scene)
        renderer.delete()
        
        # Add label
        img = Image.fromarray(color_img)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        label = labels[i]
        text_width = len(label) * 10 + 20
        draw.rectangle([10, 10, 10 + text_width, 35], fill=(0, 0, 0, 180))
        draw.text((15, 12), label, fill=(255, 255, 255), font=font)
        
        frames.append(np.array(img))
    
    return np.concatenate(frames, axis=1)


def render_side_by_side_video(
    verts_gt: np.ndarray,
    verts_gen: np.ndarray,
    faces: np.ndarray,
    output_path: str,
    labels: tuple = ("Ground Truth", "Generated"),
    fps: int = 15,
    slowdown: int = 2,
    frame_width: int = 540,
    frame_height: int = 720,
    camera_distance: float = 3.5,
    focal_length: float = 2000,
    apply_rotation_fix: bool = True,
    trim_end_frames: bool = True,
    show_progress: bool = True,
    stabilize_motion: bool = True
) -> str:
    """
    Render side-by-side comparison video of GT vs Generated motion.
    
    Args:
        verts_gt: (T, V, 3) Ground truth vertex sequence
        verts_gen: (T, V, 3) Generated vertex sequence
        faces: (F, 3) face indices
        output_path: Path to output video file (.mp4)
        labels: Tuple of labels for (GT, Generated)
        fps: Frames per second
        slowdown: Factor to slow down video (each frame repeated this many times)
        frame_width: Width of each mesh frame
        frame_height: Height of each mesh frame
        camera_distance: Distance from camera to subject
        focal_length: Camera focal length
        apply_rotation_fix: Apply 180-degree rotation fix for correct orientation
        trim_end_frames: Trim last few frames to avoid artifacts
        show_progress: Show rendering progress
        stabilize_motion: Stabilize both sequences around a fixed center
    
    Returns:
        Path to output video
    """
    if not ensure_pyrender():
        raise RuntimeError("PyRender not available for video rendering")
    
    try:
        import imageio
    except ImportError:
        os.system("pip install -q imageio[ffmpeg]")
        import imageio
    
    # Make copies to avoid modifying original
    verts_gt = verts_gt.copy()
    verts_gen = verts_gen.copy()
    
    # FIX 1: Apply 180-degree rotation around X-axis
    if apply_rotation_fix:
        verts_gt[..., 1:] *= -1
        verts_gen[..., 1:] *= -1
    
    # FIX 2: Trim end frames
    T_total = min(verts_gt.shape[0], verts_gen.shape[0])
    if trim_end_frames and T_total > 10:
        trim_amount = min(8, int(T_total * 0.15))
        T = T_total - trim_amount
        if show_progress:
            print(f"  Trimming last {trim_amount} frames to remove end-of-sequence artifacts.")
    else:
        T = T_total
    
    # Compute fixed camera target from first GT frame (after rotation)
    # CRITICAL: Do NOT recenter vertices - just use fixed_center for camera positioning
    # This matches the working approach in compare_vqvae.py and hugging_face_app.py
    fixed_center = verts_gt[0].mean(axis=0)
    
    total_video_frames = T * slowdown
    duration_sec = total_video_frames / fps
    
    if show_progress:
        print(f"  Rendering {T} motion frames -> {total_video_frames} video frames")
        print(f"  FPS: {fps}, Duration: {duration_sec:.1f}s")
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Render all frames
    frames = []
    for t in range(T):
        if show_progress and t % 10 == 0:
            print(f"    Frame {t+1}/{T}...")
        
        try:
            frame = render_side_by_side_frame(
                verts_gt[t],
                verts_gen[t],
                faces,
                labels=labels,
                fixed_center=fixed_center,
                camera_distance=camera_distance,
                focal_length=focal_length,
                frame_width=frame_width,
                frame_height=frame_height
            )
            
            # Apply slowdown
            for _ in range(slowdown):
                frames.append(frame)
        except Exception as e:
            if show_progress:
                print(f"    Error rendering frame {t}: {e}")
            break
    
    # Write video
    if len(frames) > 0:
        if show_progress:
            print(f"  Writing video to: {output_path}")
        
        if output_path.endswith('.gif'):
            imageio.mimsave(output_path, frames, fps=fps)
        else:
            imageio.mimsave(output_path, frames, fps=fps, codec='libx264', quality=8)
        
        if show_progress:
            print(f"  Video saved: {output_path} ({duration_sec:.1f}s)")
    else:
        print("  Warning: No frames rendered.")
    
    return output_path


# =====================================================================
# Interactive 3D Viewer (Open3D-based)
# =====================================================================
def ensure_open3d():
    """Install Open3D if not available"""
    try:
        import open3d as o3d
        return True
    except ImportError:
        print("Installing Open3D for interactive viewing...")
        os.system("pip install -q open3d")
        try:
            import open3d as o3d
            return True
        except ImportError:
            print("Could not install Open3D.")
            return False


def interactive_viewer(verts_seq, faces, fps=15, title="Motion Viewer"):
    """
    Launch an interactive 3D viewer with playback controls.
    
    Controls:
        - Left mouse drag: Rotate view
        - Right mouse drag / scroll: Zoom
        - Middle mouse drag: Pan
        - Space: Play/Pause animation
        - Left/Right arrows: Previous/Next frame
        - R: Reset view
        - Q: Quit
    
    Args:
        verts_seq: (T, V, 3) vertex sequence
        faces: (F, 3) face indices
        fps: Playback speed
        title: Window title
    """
    if not ensure_open3d():
        print("Open3D not available. Use --render-mode html for interactive HTML view.")
        return
    
    import open3d as o3d
    
    T = verts_seq.shape[0]
    print(f"\nLaunching interactive viewer with {T} frames...")
    print("Controls:")
    print("  Mouse drag: Rotate | Scroll: Zoom | Middle drag: Pan")
    print("  Space: Play/Pause | Arrows: Prev/Next frame")
    print("  R: Reset view | Q: Quit")
    
    # Apply 180-degree rotation (same as PyRender)
    rot_matrix = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    verts_rotated = np.array([np.dot(v, rot_matrix[:3, :3].T) for v in verts_seq])
    
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_rotated[0])
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    
    # Set mesh color (cream white like SOKE)
    mesh.paint_uniform_color([0.95, 0.92, 0.88])
    
    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title, width=1280, height=960)
    vis.add_geometry(mesh)
    
    # Set up rendering options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.95, 0.95, 0.97])
    opt.light_on = True
    opt.mesh_show_back_face = True
    
    # Animation state
    state = {
        'frame': 0,
        'playing': True,
        'last_time': 0
    }
    frame_duration = 1.0 / fps
    
    def update_mesh(frame_idx):
        """Update mesh to show specific frame"""
        mesh.vertices = o3d.utility.Vector3dVector(verts_rotated[frame_idx])
        mesh.compute_vertex_normals()
        vis.update_geometry(mesh)
        state['frame'] = frame_idx
    
    def toggle_play(vis):
        """Space: Toggle play/pause"""
        state['playing'] = not state['playing']
        status = "Playing" if state['playing'] else "Paused"
        print(f"  {status} at frame {state['frame']+1}/{T}")
        return False
    
    def next_frame(vis):
        """Right arrow: Next frame"""
        state['playing'] = False
        new_frame = (state['frame'] + 1) % T
        update_mesh(new_frame)
        print(f"  Frame {new_frame+1}/{T}")
        return False
    
    def prev_frame(vis):
        """Left arrow: Previous frame"""
        state['playing'] = False
        new_frame = (state['frame'] - 1) % T
        update_mesh(new_frame)
        print(f"  Frame {new_frame+1}/{T}")
        return False
    
    def reset_view(vis):
        """R: Reset camera view"""
        vis.reset_view_point(True)
        print("  View reset")
        return False
    
    # Register key callbacks
    vis.register_key_callback(ord(' '), toggle_play)    # Space
    vis.register_key_callback(262, next_frame)          # Right arrow
    vis.register_key_callback(263, prev_frame)          # Left arrow  
    vis.register_key_callback(ord('R'), reset_view)     # R
    vis.register_key_callback(ord('r'), reset_view)     # r
    
    # Set initial camera to frontal view
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])  # Look from front
    ctr.set_up([0, -1, 0])     # Y is up
    ctr.set_zoom(0.5)          # Zoom out to see full body
    
    # Animation loop
    import time
    print("\nAnimation started. Press Q to quit.")
    
    while True:
        current_time = time.time()
        
        # Update animation if playing
        if state['playing']:
            if current_time - state['last_time'] >= frame_duration:
                new_frame = (state['frame'] + 1) % T
                update_mesh(new_frame)
                state['last_time'] = current_time
        
        # Update visualization
        vis.poll_events()
        vis.update_renderer()
        
        # Check if window was closed
        if not vis.poll_events():
            break
    
    vis.destroy_window()
    print("Viewer closed.")


# =====================================================================
# Main Visualization Pipeline
# =====================================================================
def visualize(
    tokens,
    vqvae_ckpt=VQVAE_CHECKPOINT,
    stats_path=STATS_PATH,
    smplx_dir=SMPLX_MODEL_DIR,
    output_path=None,
    title="Generated Motion",
    fps=15,
    render_mode='html',
    style='default',
    img_size=(1080, 1080),
    slowdown=2,
    zoom=1.0,
    data_level=DEFAULT_DATA_LEVEL,
    apply_rotation_fix=True,
    trim_end_frames=True,
    lock_trans=True
):
    """
    Complete visualization pipeline: tokens -> vertices -> animation.
    
    Args:
        tokens: Motion tokens (string or list of ints)
        vqvae_ckpt: Path to VQ-VAE checkpoint
        stats_path: Path to normalization stats
        smplx_dir: Path to SMPL-X model directory
        output_path: Path to save output (HTML, MP4, or directory for frames)
        title: Animation title
        fps: Frames per second (default: 15 for sign language)
        render_mode: 
            - 'html': Interactive Plotly (web browser)
            - 'video': High-quality MP4/GIF using PyRender
            - 'frames': Individual PNG images
            - 'interactive': Real-time 3D viewer with rotation/zoom (Open3D)
        style: For video/frames - 'default', 'silhouette', 'hands-highlight'
        img_size: For video/frames - output image dimensions (width, height)
        slowdown: Factor to slow down video (2 = half speed). Only for video mode.
        zoom: Zoom level (0.5 = zoomed out/full body, 1.0 = default, 2.0 = close-up)
        data_level: "word" for word-level data, "sentence" for sentence-level (How2Sign)
                   This affects how SMPL-X parameters are mapped.
        apply_rotation_fix: Apply 180-degree rotation fix for correct orientation (video only)
        trim_end_frames: Trim last few frames to avoid artifacts (video only)
        lock_trans: Lock global translation to 0 (stabilizes visualization)
    
    Returns:
        Plotly figure (html mode), output path (video/frames), or None (interactive)
    """
    print("="*60)
    print("Motion Visualization Pipeline")
    print("="*60)
    print(f"Render mode: {render_mode}, Style: {style}, Data level: {data_level}")
    
    # Parse tokens
    print("\n[1/5] Parsing tokens...")
    token_list = parse_motion_tokens(tokens)
    print(f"   Parsed {len(token_list)} tokens")
    if not token_list:
        print("No tokens to visualize")
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
        print("No frames produced from decoder")
        return None
    
    # Convert to vertices (with data_level for correct parameter mapping)
    print(f"   Converting parameters to vertices (data_level={data_level})...")
    verts, faces = params_to_vertices(params, smplx_model, batch_size=32, data_level=data_level, lock_trans=lock_trans)
    print(f"   Vertices shape: {verts.shape}, Faces: {faces.shape}")
    
    # Render based on mode
    if render_mode == 'video':
        # High-quality video rendering using PyRender
        if output_path is None:
            output_path = os.path.join(OUTPUT_DIR, "motion_animation.mp4")
        
        print(f"\n   Rendering high-quality video ({style} style)...")
        result = render_video_pyrender(
            verts, faces, output_path,
            fps=fps,
            img_size=img_size,
            style=style,
            slowdown=slowdown,
            zoom=zoom,
            apply_rotation_fix=apply_rotation_fix,
            trim_end_frames=trim_end_frames
        )
        
    elif render_mode == 'frames':
        # Render individual frames
        if output_path is None:
            output_path = os.path.join(OUTPUT_DIR, "motion_frames")
        
        print(f"\n   Rendering individual frames ({style} style)...")
        result = render_frames_pyrender(
            verts, faces, output_path,
            img_size=img_size,
            style=style
        )
    
    elif render_mode == 'interactive':
        # Real-time 3D viewer with mouse controls (Open3D)
        print("\n   Launching interactive 3D viewer...")
        interactive_viewer(verts, faces, fps=fps, title=title)
        result = None  # No file output
        
    else:
        # Interactive HTML (Plotly) - default
        if output_path is None:
            output_path = os.path.join(OUTPUT_DIR, "motion_animation.html")
        
        print("   Creating interactive animation...")
        result = animate_motion(verts, faces, title=title, output_path=output_path, fps=fps)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    
    return result


# Legacy function for backward compatibility
def visualize_html(tokens, output_html=None, data_level="word", **kwargs):
    """
    Backward-compatible wrapper for HTML visualization.
    
    NOTE: Default data_level is "word" for backward compatibility with train_pipeline.py.
    For hybrid/sentence data, explicitly pass data_level="sentence".
    """
    return visualize(tokens, output_path=output_html, render_mode='html', data_level=data_level, **kwargs)


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
        help="Path to save output (HTML, MP4, GIF, or directory for frames)"
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
        default=15,
        help="Frames per second for animation (default: 15 for smooth sign language viewing)"
    )
    
    # Data level option (CRITICAL for correct parameter mapping)
    parser.add_argument(
        "--data-level",
        type=str,
        default=DEFAULT_DATA_LEVEL,
        choices=["word", "sentence"],
        help="Data level for SMPL-X parameter mapping:\n"
             "  word: Word-level data (original word-level VQ-VAE format)\n"
             "  sentence: Sentence-level data (How2Sign/hybrid format, default)\n"
             "This affects how the 182 SMPL-X parameters are interpreted."
    )
    
    # Rendering options
    parser.add_argument(
        "--render-mode",
        type=str,
        default="html",
        choices=["html", "video", "frames", "interactive"],
        help="Render mode:\n"
             "  html: Interactive Plotly in web browser (default)\n"
             "  video: High-quality MP4/GIF using PyRender\n"
             "  frames: Individual PNG images\n"
             "  interactive: Real-time 3D viewer with rotation/zoom controls (Open3D)"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="default",
        choices=["default", "silhouette", "hands-highlight", "wireframe-overlay"],
        help="Rendering style for video/frames mode:\n"
             "  default: Clean cream/white material (like SOKE paper)\n"
             "  silhouette: Dark silhouette (good for publications)\n"
             "  hands-highlight: Neutral body with highlighted hands\n"
             "  wireframe-overlay: Solid mesh with wireframe edges"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[1080, 1080],
        metavar=("WIDTH", "HEIGHT"),
        help="Output image size for video/frames mode (default: 1080 1080)"
    )
    parser.add_argument(
        "--slowdown",
        type=int,
        default=2,
        help="Slowdown factor for video mode (2 = half speed, 3 = third speed). "
             "Each motion frame is repeated this many times. Default: 2"
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Zoom level for video mode (0.5 = zoomed out/full body, 1.0 = default, "
             "2.0 = zoomed in/upper body). Default: 1.0"
    )
    
    # Rendering fixes
    parser.add_argument(
        "--no-rotation-fix",
        action="store_true",
        help="Disable 180-degree rotation fix (for video mode)"
    )
    parser.add_argument(
        "--no-trim-end",
        action="store_true",
        help="Disable trimming of end frames (for video mode)"
    )
    parser.add_argument(
        "--unlock-trans",
        action="store_true",
        help="Unlock global translation (may cause jerking/instability)"
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
        output_path=args.output,
        title=args.title,
        fps=args.fps,
        render_mode=args.render_mode,
        style=args.style,
        img_size=tuple(args.img_size),
        slowdown=args.slowdown,
        zoom=args.zoom,
        data_level=args.data_level,
        apply_rotation_fix=not args.no_rotation_fix,
        trim_end_frames=not args.no_trim_end,
        lock_trans=not args.unlock_trans
    )


if __name__ == "__main__":
    main()