"""DEQ-based predictor for JEPA world models.

Replaces stacked predictor blocks (e.g. 6× Transformer) with a single block
iterated to a fixed point via a deep equilibrium solver.  The DEQ finds the
equilibrium hidden state z* satisfying  z* = f(z*; c)  where f is a
ConditionalBlock gated by action embeddings and the solver is Broyden's method
with implicit-differentiation backward (O(1) memory in "depth").
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from torchdeq import get_deq
from module import modulate, FeedForward


class DEQConditionalBlock(nn.Module):
    """Single conditional block designed for DEQ fixed-point iteration.

    Architecturally mirrors ConditionalBlock (Attention + MLP with AdaLN-zero
    gating), but wraps the output gates in ``tanh`` so the function stays
    near-identity during early training — critical for DEQ convergence.
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        """Apply one block iteration.

        Args:
            x: (B, T, D) current hidden state.
            c: (B, T, D) conditioning (action embeddings).
        Returns:
            (B, T, D) next iterate.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        gate_msa = gate_msa.tanh()
        gate_mlp = gate_mlp.tanh()

        x = x + gate_msa * self._attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def _attn(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class DEQFixpointFunc(nn.Module):
    """Wrapper that adapts a conditional block to the ``f(z) = z + block(z, c)``
    signature expected by the DEQ solver.

    The residual structure ``f(z) = z + g(z)`` (where ``g`` is the block)
    makes the fixed point problem well-posed: at equilibrium, ``g(z*) = 0``.
    With zero-initialised gates, ``g`` is identically zero at the start of
    training so the DEQ converges in 1 iteration.
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        self.c = None

    def forward(self, z):
        return z + self.block(z, self.c)


class DEQPredictor(nn.Module):
    """Autoregressive next-step predictor based on a deep-equilibrium block.

    Usage mirrors ``ARPredictor`` so it slots directly into the existing
    ``JEPA`` model — just swap ``predictor._target_`` in the Hydra config.

    Args:
        num_frames: Max sequence length (for positional embeddings).
        input_dim: Dimension of input embeddings.
        hidden_dim: Dimension of the DEQ hidden state.
        output_dim: Dimension of output predictions (usually same as input_dim).
        depth: *Ignored for DEQ* (the DEQ replaces explicit depth).
        heads: Number of attention heads.
        mlp_dim: Hidden dimension of the MLP sub-layer.
        dim_head: Dimension per head.
        dropout: Dropout probability.
        emb_dropout: Dropout applied after positional embeddings.
        f_max_iter: Max solver iterations during training.
        eval_f_max_iter: Max solver iterations during evaluation.
        f_solver: Forward solver (``'broyden'``, ``'anderson'``, etc.).
        b_solver: Backward solver (``'broyden'``, ``'none'``, etc.).
        f_tol: Solver tolerance for forward pass.
        core: DEQ core type (``'sliced'`` or ``'indexing'``).
    """

    def __init__(
        self,
        *,
        num_frames,
        input_dim,
        hidden_dim,
        output_dim=None,
        depth=1,
        heads=8,
        mlp_dim=2048,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.0,
        f_max_iter=25,
        eval_f_max_iter=40,
        f_solver="broyden",
        b_solver="broyden",
        f_tol=1e-5,
        core="sliced",
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)

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

        block = DEQConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
        self.func = DEQFixpointFunc(block)

        self.deq = get_deq(
            core=core,
            f_max_iter=f_max_iter,
            eval_f_max_iter=eval_f_max_iter,
            f_solver=f_solver,
            b_solver=b_solver,
            f_tol=f_tol,
        )

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim or input_dim)
            if (output_dim or input_dim) != hidden_dim
            else nn.Identity()
        )

    def forward(self, x, c):
        """
        Args:
            x: (B, T, input_dim) context embeddings.
            c: (B, T, input_dim) action embeddings.
        Returns:
            (B, T, output_dim) predicted next-step embeddings.
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)

        x = self.input_proj(x)
        c_proj = self.cond_proj(c)

        self.func.c = c_proj
        z_star, _ = self.deq(self.func, x)

        out = self.norm(z_star[0])
        out = self.output_proj(out)
        return out
