"""Render low-dim state vectors to images for the image modality.

Each renderer maps (state, action, prev_frame) -> (H, W, 3) uint8.
"""

import numpy as np


def render_state_to_image(state, action, task_name, H=64, W=64, prev_frame=None):
    if task_name == "linear_ar":
        return _render_gaussian_blob(state, H, W, prev_frame)
    elif task_name == "nback":
        return _render_token_grid(state, H, W)
    elif task_name == "delayed_copy":
        return _render_pattern_grid(state, H, W)
    elif task_name == "slowfast":
        return _render_trail_blob(state, H, W, prev_frame)
    elif task_name == "chaotic":
        return _render_streamline(state, H, W)
    else:
        return _render_gaussian_blob(state, H, W, prev_frame)


def _normalize_state(state, dims=2):
    """Take first dims*2 values from state, normalize to [-1, 1]."""
    vals = state[:dims * 2]
    max_abs = max(np.abs(vals).max(), 1e-8)
    return vals / max_abs


def _render_gaussian_blob(state, H, W, prev_frame=None):
    """Two Gaussian blobs whose positions are controlled by state[:4]."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    coords = _normalize_state(state, 2)
    def add_blob(cx, cy, color, sigma=0.08):
        yy, xx = np.mgrid[:H, :W]
        cy_pix = (cy * 0.4 + 0.5) * H
        cx_pix = (cx * 0.4 + 0.5) * W
        g = np.exp(-((xx - cx_pix) ** 2 + (yy - cy_pix) ** 2) / (2 * (sigma * H) ** 2))
        for c in range(3):
            img[:, :, c] = np.clip(img[:, :, c].astype(float) + g * color[c], 0, 255).astype(np.uint8)
    add_blob(coords[0], coords[1], (180, 60, 60))
    if len(coords) >= 4:
        add_blob(coords[2], coords[3], (60, 120, 200), sigma=0.05)
    return img


def _render_token_grid(state, H, W):
    """Render discrete tokens as colored cells in a grid."""
    vocab_size = min(state.shape[0], 16)
    grid_size = int(np.ceil(np.sqrt(vocab_size)))
    cell_h, cell_w = H // grid_size, W // grid_size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 128, 0),
              (128, 0, 255), (0, 128, 255), (255, 255, 128), (128, 255, 128),
              (255, 128, 128), (128, 128, 255), (0, 0, 0), (255, 255, 255)]
    for i in range(vocab_size):
        r, c = divmod(i, grid_size)
        val = state[i]
        if val > 0.5:
            y1, x1 = r * cell_h, c * cell_w
            y2, x2 = (r + 1) * cell_h, (c + 1) * cell_w
            img[y1:y2, x1:x2] = colors[i % len(colors)]
    return img


def _render_pattern_grid(state, H, W):
    """Render prompt vs recall pattern as two-color field."""
    vocab_size = min(state.shape[0], 8)
    return _render_token_grid(state, H, W)


def _render_trail_blob(state, H, W, prev_frame=None):
    """Blob with fading trail for slow + fast."""
    if prev_frame is None:
        img = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        img = (prev_frame.astype(float) * 0.85).astype(np.uint8)
    coords = _normalize_state(state, 1)
    cx = (coords[0] * 0.4 + 0.5) * W
    if len(coords) >= 2:
        cy = (coords[1] * 0.4 + 0.5) * H
    else:
        cy = H // 2
    yy, xx = np.mgrid[:H, :W]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (0.06 * H) ** 2))
    img[:, :, 0] = np.clip(img[:, :, 0].astype(float) + g * 200, 0, 255).astype(np.uint8)
    # Fast component as a small bright dot
    if len(coords) >= 4:
        fx = (coords[2] * 0.3 + 0.5) * W
        fy = (coords[3] * 0.3 + 0.5) * H
        g2 = np.exp(-((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * (0.02 * H) ** 2))
        img[:, :, 1] = np.clip(img[:, :, 1].astype(float) + g2 * 255, 0, 255).astype(np.uint8)
    return img


def _render_streamline(state, H, W):
    """Render Lorenz-like state as a colored radial pattern."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    v = _normalize_state(state, 3)
    cx = (v[0] * 0.3 + 0.5) * W
    cy = (v[1] * 0.3 + 0.5) * H
    yy, xx = np.mgrid[:H, :W]
    dx, dy = xx - cx, yy - cy
    mag = np.sqrt(dx ** 2 + dy ** 2) / (H * 0.3)
    angle = np.arctan2(dy, dx) / np.pi
    r = np.clip(1 - mag, 0, 1)
    hue = (angle * 0.5 + 0.5).astype(float)
    img[:, :, 0] = np.clip(r * 255 * (1 - hue * 0.5), 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(r * 255 * (0.3 + hue * 0.4), 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip(r * 255 * (0.5 + hue * 0.3), 0, 255).astype(np.uint8)
    return img
