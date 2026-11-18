"""
Stage 1: 字符级莎士比亚 (TinyShakespeare) 全光 NanoGPT 实验
============================================================
目标:
  验证全光 Native Attention (sin^4 偏振门) 在语言建模任务上的梯度传播与学习能力。
评判标准:
  - 初始时输出为随机乱码 (Loss ~ 4.18)
  - 训练 500~1000 步后，Loss 稳定下降至 2.0 左右，模型开始生成标准剧本结构与伪英语词汇！
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from model import OpticalNanoGPT, GPTConfig, audit_optical_pipeline
from data.prepare_shakespeare import download_data, FILE_PATH

def train_shakespeare(attn_type='optical', pure_optical=False, max_iters=1000, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print("=" * 80)
    print(f"   🎭 Stage 1: 字符级莎士比亚全光 NanoGPT 训练 (Attention: {attn_type.upper()}) 🎭")
    print(f"   硬件加速设备: {device.upper()} (GPU 极速训练)")
    if device == 'cuda':
        print(f"   GPU 型号: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # 1. 确保数据就绪
    download_data()
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"数据集大小: {len(text):,} 字符 | 字符级词表大小: {vocab_size}")

    # 简易字符编解码器
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # 数据集切分
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    block_size = 64
    batch_size = 32

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # 2. 初始化全光 NanoGPT 模型
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
        attention_type=attn_type,
        normalize_beam=not pure_optical,           # 若 pure_optical=True，彻底关闭动态除法求和！
        mlp_type='optical' if pure_optical else 'gelu', # 若 pure_optical=True，MLP 全换成偏振门！
        norm_type='none' if pure_optical else 'layernorm' # 若 pure_optical=True，干掉 LayerNorm！
    )
    audit_optical_pipeline(config)
    model = OpticalNanoGPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e3:.1f} K")
    print(f"注意力机制: {'✨ 全光 sin^4 偏振门 (Lyot 门)' if attn_type == 'optical' else '⚡ 传统电子 Softmax'}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # 3. 初始未训练时的乱码生成
    print("\n" + "─" * 80)
    print("【初始状态 (未训练) 文本生成测试】:")
    start_prompt = "ROMEO:\n"
    context = torch.tensor([encode(start_prompt)], dtype=torch.long, device=device)
    raw_gen = decode(model.generate(context, max_new_tokens=80, temperature=0.8)[0].tolist())
    print(raw_gen)
    print("─" * 80 + "\n")

    # 4. 训练循环
    t0 = time.time()
    for iter_step in range(1, max_iters + 1):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter_step % 200 == 0 or iter_step == max_iters:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch('val')
                _, val_loss = model(xv, yv)
            model.train()
            
            # 读取当前光学旋钮 (optical scale) 的自适应状态
            if attn_type == 'optical':
                scales = [b.attn.optical_scale.item() for b in model.transformer.h]
                scale_str = f" | 光学旋钮 Phase Scales: [{scales[0]:.3f}, {scales[1]:.3f}...]"
            else:
                scale_str = ""

            t_elapsed = time.time() - t0
            print(f"Step {iter_step:4d}/{max_iters} | 训练损失: {loss.item():.4f} | 验证损失: {val_loss.item():.4f} | 耗时: {t_elapsed:.1f}s{scale_str}")

    # 5. 训练完成后的生成测试
    print("\n" + "=" * 80)
    print("【训练后 (偏振门生效) 文本生成实测】:")
    print("=" * 80)
    model.eval()
    with torch.no_grad():
        context = torch.tensor([encode("ROMEO:\n")], dtype=torch.long, device=device)
        out_tokens = model.generate(context, max_new_tokens=250, temperature=0.75, top_k=10)
        generated_text = decode(out_tokens[0].tolist())
        print(generated_text)
    print("=" * 80)

    # 保存检查点
    ckpt_path = f"optical_shakespeare_{attn_type}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"✅ 模型检查点已保存至: {ckpt_path}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1: TinyShakespeare Optical NanoGPT")
    parser.add_argument("--attn-type", type=str, default="optical", choices=["optical", "softmax"], help="Attention type")
    parser.add_argument("--pure-optical", action="store_true", default=True, help="Use pure optical mode (no LayerNorm, no sum division, sin^4 MLP)")
    parser.add_argument("--with-layernorm", dest="pure_optical", action="store_false", help="Enable electronic LayerNorm and beam sum normalization")
    parser.add_argument("--iters", type=int, default=800, help="Training iterations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    train_shakespeare(
        attn_type=args.attn_type,
        pure_optical=args.pure_optical,
        max_iters=args.iters,
        device=args.device
    )
