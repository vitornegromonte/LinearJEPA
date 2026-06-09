import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        block_class=ConditionalBlock,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=block_class,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x

##################################################################################

class DeltaNetAttention(nn.Module):
    """Linear attention via delta rule (Yang et al., 2024). O(T) instead of O(T²).

    Uses fla.layers.DeltaNet (CUDA kernel) when available, pure PyTorch fallback.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self._use_optimized = False
        try:
            from fla.layers import DeltaNet as _DeltaNet
            self.delta = _DeltaNet(d_model=dim, n_head=heads, head_dim=dim_head)
            self._use_optimized = True
        except Exception:
            pass

        if not self._use_optimized:
            inner_dim = dim_head * heads
            self.heads = heads
            self.norm = nn.LayerNorm(dim)
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim), nn.Dropout(dropout)
            )

    def forward(self, x, causal=True):
        if self._use_optimized:
            return self.delta(x)

        B, T, D = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)

        H, D_h = q.shape[1], q.shape[-1]
        S = torch.zeros(B, H, D_h, D_h, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            q_t = q[:, :, t:t+1]
            k_t = k[:, :, t:t+1]
            v_t = v[:, :, t:t+1]

            k_attn = (S @ k_t.transpose(-1, -2))
            delta = v_t.transpose(-1, -2) - k_attn
            S = S + (delta @ k_t)

            y_t = (S @ q_t.transpose(-1, -2)).transpose(-1, -2)
            outputs.append(y_t)

        out = torch.cat(outputs, dim=-2)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class DeltaNetConditionalBlock(nn.Module):
    """Transformer block with linear attention (DeltaNet) + AdaLN-zero conditioning."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = DeltaNetAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class MambaBlock(nn.Module):
    """Mamba SSM block with dual backend: optimized (mamba_ssm) or pure PyTorch fallback."""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = dim * expand
        self._use_optimized = False

        try:
            from mamba_ssm import Mamba as MambaSSM
            self.mamba = MambaSSM(
                d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand,
            )
            self._use_optimized = True
        except ImportError:
            pass

        if not self._use_optimized:
            self.norm = nn.LayerNorm(dim)
            self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(
                self.d_inner, self.d_inner, d_conv,
                groups=self.d_inner, padding=d_conv - 1,
            )
            self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
            self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
            self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
            self.D = nn.Parameter(torch.ones(self.d_inner))
            self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
            self.act = nn.SiLU()

    def forward(self, x):
        if self._use_optimized:
            return self.mamba(x)

        B, L, D = x.shape
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        x_proj = rearrange(x_proj, "b l d -> b d l")
        x_proj = self.conv1d(x_proj)[..., :L]
        x_proj = rearrange(x_proj, "b d l -> b l d")
        x_proj = self.act(x_proj)

        dt_params = self.x_proj(x_proj)
        B_proj, C, dt = dt_params.split([self.d_state, self.d_state, self.d_inner], dim=-1)
        dt = self.dt_proj(dt)
        dt = F.softplus(dt)

        A = -torch.exp(self.A_log)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            dt_t = dt[:, t]
            B_t = B_proj[:, t]
            C_t = C[:, t]
            x_t = x_proj[:, t]

            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            y_t = torch.einsum('bnd,bd->bn', h, C_t) + self.D * x_t
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        y = y * self.act(z)
        return self.out_proj(y) + residual

    def step(self, x, state=None):
        B = x.shape[0]
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        x_proj = rearrange(x_proj, "b 1 d -> b d 1")

        if state is None:
            conv_buffer = torch.zeros(
                B, self.d_inner, self.d_conv - 1, device=x.device, dtype=x.dtype
            )
            ssm_h = torch.zeros(
                B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype
            )
        else:
            conv_buffer = state["conv_buffer"]
            ssm_h = state["ssm_h"]

        full_conv = torch.cat([conv_buffer, x_proj], dim=-1)
        new_buffer = full_conv[:, :, -(self.d_conv - 1):]
        x_proj = self.conv1d(full_conv)[..., -1:]
        x_proj = self.act(rearrange(x_proj, "b d 1 -> b 1 d"))

        dt_params = self.x_proj(x_proj)
        B_proj, C, dt = dt_params.split([self.d_state, self.d_state, self.d_inner], dim=-1)
        dt = self.dt_proj(dt)
        dt = F.softplus(dt)

        A = -torch.exp(self.A_log)
        x_proj_sq = x_proj.squeeze(1)
        dt_sq = dt.squeeze(1)
        B_proj_sq = B_proj.squeeze(1)
        C_sq = C.squeeze(1)

        A_bar = torch.exp(dt_sq.unsqueeze(-1) * A.unsqueeze(0))
        B_bar = dt_sq.unsqueeze(-1) * B_proj_sq.unsqueeze(1)
        ssm_h = A_bar * ssm_h + B_bar * x_proj_sq.unsqueeze(-1)
        y_t = torch.einsum('bnd,bd->bn', ssm_h, C_sq) + self.D * x_proj_sq
        y_t = y_t.unsqueeze(1) * self.act(z)
        y_t = self.out_proj(y_t) + residual

        return y_t, {"conv_buffer": new_buffer, "ssm_h": ssm_h}


class MambaPredictor(nn.Module):
    """Autoregressive predictor using stacked Mamba blocks with action conditioning."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        input_dim,
        hidden_dim,
        output_dim=None,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            MambaBlock(hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim or input_dim)
            if (output_dim or input_dim) != hidden_dim
            else nn.Identity()
        )

    def forward(self, x, c):
        x = self.input_proj(x)
        x = x + self.cond_proj(c)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.output_proj(x)

    def get_init_state(self, x, c):
        x = self.input_proj(x)
        x = x + self.cond_proj(c)
        states = []
        for block in self.blocks:
            B, L, D = x.shape
            residual = x
            x = block.norm(x)
            xz = block.in_proj(x)
            x_proj, z = xz.chunk(2, dim=-1)
            x_proj = rearrange(x_proj, "b l d -> b d l")
            x_proj = block.conv1d(x_proj)[..., :L]
            x_proj = rearrange(x_proj, "b d l -> b l d")
            x_proj = block.act(x_proj)

            dt_params = block.x_proj(x_proj)
            B_proj, C, dt = dt_params.split([block.d_state, block.d_state, block.d_inner], dim=-1)
            dt = block.dt_proj(dt)
            dt = F.softplus(dt)

            A = -torch.exp(block.A_log)
            h = torch.zeros(B, block.d_inner, block.d_state, device=x.device, dtype=x.dtype)
            for t in range(L):
                dt_t = dt[:, t]
                B_t = B_proj[:, t]
                x_t = x_proj[:, t]
                A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
                B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
                h = A_bar * h + B_bar * x_t.unsqueeze(-1)

            C_last = C[:, -1]
            y_final = torch.einsum('bnd,bd->bn', h, C_last) + block.D * x_proj[:, -1]
            x_out = y_final.unsqueeze(1) * block.act(z[:, -1:])
            x = block.out_proj(x_out) + residual

            if L < block.d_conv - 1:
                conv_buffer = torch.zeros(B, block.d_inner, block.d_conv - 1, device=x.device, dtype=x.dtype)
            else:
                conv_buffer = rearrange(x_proj[:, -(block.d_conv - 1):], "b l d -> b d l")
            states.append({"conv_buffer": conv_buffer, "ssm_h": h.clone()})

        x = self.norm(x)
        return self.output_proj(x), states

    def step(self, x, c, state):
        x = self.input_proj(x)
        x = x + self.cond_proj(c)
        new_states = []
        for i, block in enumerate(self.blocks):
            block_state_i = state[i] if state is not None else None
            x, block_state = block.step(x, block_state_i)
            new_states.append(block_state)
        x = self.norm(x)
        return self.output_proj(x), new_states
