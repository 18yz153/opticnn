"""
NanoGPT 全光架构核心模型 (model.py)
=====================================
在 Andrej Karpathy 的 NanoGPT 极简架构基础上，
将电子域的 Softmax 注意力机制升级为【全光原生偏振门 (Polarization Gate) 注意力机制】:
  - 传统电子 Attention: att = F.softmax(att, dim=-1)
  - 全光 Native Attention:
      self.optical_scale = nn.Parameter(torch.ones(1) * 0.1) # 可学习光学相位旋钮
      theta = att * self.optical_scale                        # 偏振偏转角
      att = torch.sin(theta) ** 4                             # 两级偏振片 (Lyot 门) 透过率
      att = att.masked_fill(mask == 0, 0.0)                   # 物理光束阻断 (透射为 0)
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 128      # 最大上下文长度
    vocab_size: int = 65       # 字符表大小
    n_layer: int = 4           # Transformer 层数
    n_head: int = 4            # 注意力头数
    n_embd: int = 128          # 嵌入维度
    dropout: float = 0.0
    bias: bool = False
    # 注意力模式: 'optical' (全光 sin^4 偏振门) 或 'softmax' (传统电子基线)
    attention_type: str = 'optical'
    # 是否进行无损光束总功率归一化 (若为 False, 则采用完全被动的无除法固定耦合衰减 1/sqrt(T))
    normalize_beam: bool = True
    # MLP 激活函数模式: 'optical' (sin^4 偏振透射) 或 'gelu' (传统电子激活)
    mlp_type: str = 'optical'
    # 归一化模式: 'layernorm' (电子 LayerNorm) 或 'none' (纯光学定向耦合直通)
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

        # 核心 Code Hack: 光学旋钮 (可学习参数，动态适配电光调制相位区间)
        self.optical_scale = nn.Parameter(torch.ones(1) * 0.1)

        # 因果遮罩 (下三角阵)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 缩放点积: (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.attention_type == 'softmax':
            # 传统电子 Attention
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
        elif self.attention_type == 'optical':
            # 全光 Native 偏振门 Attention:
            # 1. 送入干涉仪并调节偏振角
            theta = att * self.optical_scale
            # 2. 穿过双折射级联偏振片 (sin^4 响应)
            att = torch.sin(theta) ** 4
            # 3. 物理因果遮罩: 未来时间步的光被不透明光阑完全阻断 (透射率精确为 0.0)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, 0.0)
            # 4. 光束能量分束:
            if self.normalize_beam:
                # 动态分束归一化 (需全局光强反馈)
                att = att / (att.sum(dim=-1, keepdim=True) + 1e-6)
            else:
                # 【纯被动无除法全光模式】: 采用波导中固定的 1xT 无源分束损耗 1/sqrt(T)，
                # 彻底消除任何动态求和除法，纯靠光波在波导中的自然传播！
                att = att * (1.0 / math.sqrt(T))

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
            # 全光原生 MLP 偏振门: 可学习相位旋钮 + sin^4 透射率
            self.mlp_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        if self.mlp_type == 'gelu':
            x = self.nonlin(x)
        elif self.mlp_type == 'optical':
            # 全光偏振门透射率: 彻底干掉电子 GELU/ReLU
            x = torch.sin(x * self.mlp_scale) ** 4
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
            # 纯光学直通 (3dB 定向耦合器无源叠加，无电子 LayerNorm)
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
        # 权重共享
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化权重
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
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
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
        """自回归生成文本"""
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
    物理审计: 逐项核查整个 NanoGPT 计算流中，哪些是真正的纯光运算，哪些偷用了电学运算。
    """
    print("\n" + "=" * 80)
    print("        🔬 NanoGPT 全计算流程『光 vs 电』真实物理保真度审计报告 🔬")
    print("=" * 80)
    print(f"{'计算环节':<18} | {'数学表达式':<24} | {'物理实现方式':<20} | {'属性判定'}")
    print("-" * 80)
    print(f"{'Q,K,V 投影矩阵':<18} | {'x @ W_q, x @ W_k, x @ W_v':<24} | {'MZI 阵列 / 硅光波导网格':<20} | {'🟢 纯光学 (皮秒级)'}")
    print(f"{'注意力点积打分':<18} | {'Q @ K^T':<24} | {'空间衍射 / 波导交叉乘积':<20} | {'🟢 纯光学 (光速)'}")
    print(f"{'因果遮罩 (Mask)':<18} | {'masked_fill(0.0)':<24} | {'物理光阑 / 空间不透明快门':<20} | {'🟢 纯光学 (物理阻断)'}")
    print(f"{'注意力非线性':<18} | {'sin^4(theta)':<24} | {'两级双折射偏振滤光片':<20} | {'🟢 纯光学 (Lyot门)'}")
    
    if config.normalize_beam:
        print(f"{'注意力归一化':<18} | {'/ sum(att)':<24} | {'软件除法 (偷用电子求和)':<20} | {'🔴 电子运算 (瓶颈)'}")
    else:
        print(f"{'注意力归一化':<18} | {'* (1/sqrt(T))':<24} | {'无源固定无耗分束器':<20} | {'🟢 纯光学 (零计算开销)'}")
        
    print(f"{'Value 加权汇聚':<18} | {'att @ V':<24} | {'多波导干涉叠加汇聚':<20} | {'🟢 纯光学 (天然加法)'}")
    print(f"{'残差跳连相加':<18} | {'x + attn(x)':<24} | {'3dB 定向耦合器合波':<20} | {'🟢 纯光学 (无源器件)'}")

    if config.norm_type == 'layernorm':
        print(f"{'LayerNorm 归一化':<18} | {'(x - mu) / sigma':<24} | {'电子算均值方差除法':<20} | {'🔴 电子运算 (瓶颈)'}")
    else:
        print(f"{'LayerNorm 归一化':<18} | {'None (直通叠加)':<24} | {'纯光路直通传输':<20} | {'🟢 纯光学 (无电子层)'}")

    if config.mlp_type == 'gelu':
        print(f"{'MLP 非线性激活':<18} | {'GELU(z)':<24} | {'传统 CPU/GPU 激活':<20} | {'🔴 电子运算 (瓶颈)'}")
    else:
        print(f"{'MLP 非线性激活':<18} | {'sin^4(z * scale)':<24} | {'偏振门 / 可饱和吸收体':<20} | {'🟢 纯光学 (物理透射)'}")

    print(f"{'Token 词表嵌入':<18} | {'Embedding(idx)':<24} | {'电子查找表 / 激光源编码':<20} | {'🟡 边界光电调制 (E/O)'}")
    print(f"{'最终输出分类':<18} | {'Argmax(detectors)':<24} | {'探测器阵列测最大光强':<20} | {'🟡 边界光电探测 (O/E)'}")
    print("=" * 80 + "\n")
