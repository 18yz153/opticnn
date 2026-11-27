"""
Optical NanoGPT Core Model (model.py)
=====================================
Based on Andrej Karpathy's NanoGPT architecture,
with electronic Softmax replaced by native optical attention mechanisms:
  - Electronic Softmax: att = F.softmax(att, dim=-1)
  - Optical Native Attention (sin^4 polarization gate):
      theta = att * self.optical_scale + self.optical_bias
      att = torch.sin(theta) ** 4
  - Saturable Absorber (SA) Attention:
      T_sa = 0.05 + 0.95 * (u / (1.0 + u))
      att = I_pos * T_sa
  - Semiconductor Optical Amplifier (SOA) Cross-Gain Modulation (XGM):
      G = G0 / (1.0 + P_total / P_sat)
      att = P_in * G  (natural physical sum normalization)
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 128      # Maximum context length
    vocab_size: int = 65       # Vocabulary size
    n_layer: int = 4           # Number of transformer layers
    n_head: int = 4            # Number of attention heads
    n_embd: int = 128          # Embedding dimension
    dropout: float = 0.0
    bias: bool = False
    # Attention type: 'optical' (sin^4), 'softmax' (electronic), 'sa' (saturable absorber), 'soa' (SOA XGM)
    attention_type: str = 'optical'
    # Normalize beam power (True = dynamic division/AGC, False = passive 1/sqrt(T) splitter)
    normalize_beam: bool = True
    # MLP activation type: 'optical' (sin^4), 'gelu' (electronic), 'sa' (saturable absorber)
    mlp_type: str = 'optical'
    # Normalization type: 'layernorm' (electronic LayerNorm) or 'none' (optical direct connection)
    norm_type: str = 'layernorm'

class OpticalCausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.attention_type = config.attention_type
        self.normalize_beam = config.normalize_beam

        # Optical learnable parameters (phase scale and DC bias)
        self.optical_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.optical_bias = nn.Parameter(torch.zeros(1))

        # Saturable absorber saturation intensity (I_sat)
        self.sa_sat = nn.Parameter(torch.ones(1) * 1.0)

        # Semiconductor optical amplifier parameters (small-signal gain G0, saturation power P_sat)
        self.soa_gain = nn.Parameter(torch.ones(1) * 2.0)
        self.soa_sat = nn.Parameter(torch.ones(1) * 1.0)

        # Causal mask (lower triangular)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                         .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Scaled dot-product: (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.attention_type == 'softmax':
            # Electronic Softmax Attention
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
        elif self.attention_type == 'optical':
            # Optical Native Polarization Gate (sin^4)
            theta = att * self.optical_scale + self.optical_bias
            att = torch.sin(theta) ** 4
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, 0.0)
            if self.normalize_beam:
                att = att / (att.sum(dim=-1, keepdim=True) + 1e-6)
            else:
                att = att * (1.0 / math.sqrt(T))
        elif self.attention_type == 'sa':
            # Saturable Absorber (SA - Pauli blocking): smooth threshold without gradient dead zone
            theta = att * self.optical_scale + self.optical_bias
            I_pos = F.relu(theta)
            u = (I_pos / (torch.abs(self.sa_sat) + 1e-4)) ** 2
            T_sa = 0.05 + 0.95 * (u / (1.0 + u))
            att = I_pos * T_sa
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, 0.0)
            if self.normalize_beam:
                att = att / (att.sum(dim=-1, keepdim=True) + 1e-6)
            else:
                att = att * (1.0 / math.sqrt(T))
        elif self.attention_type == 'soa':
            # Semiconductor Optical Amplifier (SOA XGM - Carrier Depletion Competition)
            # Physical carrier saturation naturally normalizes total output power
            P_in = F.relu(att * self.optical_scale + self.optical_bias)
            P_in = P_in.masked_fill(self.bias[:, :, :T, :T] == 0, 0.0)
            P_total = P_in.sum(dim=-1, keepdim=True)
            G = torch.abs(self.soa_gain) / (1.0 + P_total / (torch.abs(self.soa_sat) + 1e-4))
            att = P_in * G

        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.mlp_type = config.mlp_type
        if self.mlp_type == 'gelu':
            self.nonlin = nn.GELU()
        elif self.mlp_type == 'optical':
            self.mlp_scale = nn.Parameter(torch.ones(1) * 0.1)
        elif self.mlp_type == 'sa':
            self.sa_sat = nn.Parameter(torch.ones(1) * 1.0)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        if self.mlp_type == 'gelu':
            x = self.nonlin(x)
        elif self.mlp_type == 'optical':
            x = torch.sin(x * self.mlp_scale) ** 4
        elif self.mlp_type == 'sa':
            I_pos = F.relu(x)
            u = (I_pos / (torch.abs(self.sa_sat) + 1e-4)) ** 2
            T_sa = 0.05 + 0.95 * (u / (1.0 + u))
            x = I_pos * T_sa
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm_type = config.norm_type
        if self.norm_type == 'layernorm':
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = OpticalCausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        if self.norm_type == 'layernorm':
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            # Pure optical direct connection (3dB directional coupler addition, no LayerNorm)
            x = x + self.attn(x)
            x = x + self.mlp(x)
        return x

class OpticalNanoGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd) if config.norm_type == 'layernorm' else nn.Identity(),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive text generation."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def audit_optical_pipeline(config: GPTConfig):
    """
    Physical Audit: Checks whether each operation in NanoGPT is purely optical or electronic.
    """
    print("\n" + "=" * 80)
    print("        NanoGPT Optical vs Electronic Physical Fidelity Audit Report")
    print("=" * 80)
    print(f"{'Operation':<22} | {'Math Expression':<24} | {'Physical Realization':<22} | {'Fidelity'}")
    print("-" * 80)
    print(f"{'Q,K,V Projections':<22} | {'x @ W_q, x @ W_k, x @ W_v':<24} | {'MZI Mesh / Waveguides':<22} | {'[Pure Optical] (ps)'}")
    print(f"{'Dot-product Scoring':<22} | {'Q @ K^T':<24} | {'Spatial Interference':<22} | {'[Pure Optical] (c)'}")
    print(f"{'Causal Mask':<22} | {'masked_fill(0.0)':<24} | {'Physical Optical Aperture':<22} | {'[Pure Optical] (block)'}")

    if config.attention_type == 'softmax':
        print(f"{'Attention Non-lin':<22} | {'Softmax(QK^T)':<24} | {'Electronic Exp & Div':<22} | {'[Electronic Baseline]'}")
    elif config.attention_type == 'optical':
        print(f"{'Attention Non-lin':<22} | {'sin^4(theta)':<24} | {'Lyot Polarization Gate':<22} | {'[Pure Optical] (Lyot)'}")
    elif config.attention_type == 'sa':
        print(f"{'Attention Non-lin':<22} | {'T_SA(I) * I':<22} | {'Saturable Absorber':<22} | {'[Pure Optical] (Pauli)'}")
    elif config.attention_type == 'soa':
        print(f"{'Attention Non-lin':<22} | {'SOA-XGM(P)':<24} | {'SOA Carrier Competition':<22} | {'[Active Optical]'}")

    if config.attention_type == 'soa':
        print(f"{'Beam Normalization':<22} | {'1/(1 + P/P_sat)':<24} | {'SOA Carrier Saturation':<22} | {'[Active Optical] (no div)'}")
    elif config.normalize_beam:
        print(f"{'Beam Normalization':<22} | {'/ sum(att)':<24} | {'Software Division (ALU)':<22} | {'[Electronic Bottleneck]'}")
    else:
        print(f"{'Beam Normalization':<22} | {'* (1/sqrt(T))':<24} | {'Passive Waveguide Splitter':<22} | {'[Pure Optical] (zero cost)'}")

    print(f"{'Value Weighting':<22} | {'att @ V':<24} | {'Multi-waveguide Sum':<22} | {'[Pure Optical] (natural)'}")
    print(f"{'Residual Add':<22} | {'x + attn(x)':<24} | {'3dB Directional Coupler':<22} | {'[Pure Optical] (passive)'}")

    if config.norm_type == 'layernorm':
        print(f"{'LayerNorm':<22} | {'(x - mu) / sigma':<24} | {'Electronic DSP Mean/Var':<22} | {'[Electronic Bottleneck]'}")
    else:
        print(f"{'LayerNorm':<22} | {'None (direct)':<24} | {'Passive Optical Path':<22} | {'[Pure Optical] (no norm)'}")

    if config.mlp_type == 'gelu':
        print(f"{'MLP Non-linearity':<22} | {'GELU(z)':<24} | {'CPU/GPU ALU':<22} | {'[Electronic Bottleneck]'}")
    elif config.mlp_type == 'sa':
        print(f"{'MLP Non-linearity':<22} | {'T_SA(z) * z':<24} | {'Saturable Absorber':<22} | {'[Pure Optical] (SA)'}")
    else:
        print(f"{'MLP Non-linearity':<22} | {'sin^4(z * scale)':<24} | {'Polarization Gate':<22} | {'[Pure Optical] (Lyot)'}")

    print(f"{'Token Embedding':<22} | {'Embedding(idx)':<24} | {'Laser Array + Modulator':<22} | {'[Boundary E/O]'}")
    print(f"{'Output Decoding':<22} | {'Argmax(detectors)':<24} | {'Photodetector Array':<22} | {'[Boundary O/E]'}")
    print("=" * 80 + "\n")
