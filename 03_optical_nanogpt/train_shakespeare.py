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

def train_shakespeare(attn_type='optical', pure_optical=False, max_iters=1000, 
                      eval_interval=100, patience=4, min_delta=0.005, target_loss=2.45,
                      device='cuda' if torch.cuda.is_available() else 'cpu'):
    print("=" * 80)
    print(f"   🎭 Stage 1: 字符级莎士比亚全光 NanoGPT 训练 (Attention: {attn_type.upper()}) 🎭")
    print(f"   硬件加速设备: {device.upper()} (GPU 极速训练)")
    if device == 'cuda':
        print(f"   GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"   停机策略: MaxIters={max_iters} | Patience={patience} | MinDelta={min_delta} | TargetLoss={target_loss}")
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

    # 4. 智能训练循环 (Max Iters + Patience Early Stop + Target Stop)
    t0 = time.time()
    best_val_loss = float('inf')
    best_step = 0
    patience_counter = 0
    best_state = None
    stop_reason = "达到最大迭代步数 (Max Iters Reached)"

    for iter_step in range(1, max_iters + 1):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter_step % eval_interval == 0 or iter_step == max_iters:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch('val')
                _, val_loss = model(xv, yv)
            model.train()
            
            val_loss_val = val_loss.item()
            t_elapsed = time.time() - t0

            # 读取当前光学旋钮 (optical scale) 的自适应状态
            if attn_type == 'optical':
                scales = [b.attn.optical_scale.item() for b in model.transformer.h]
                scale_str = f" | 光学旋钮 Phase Scales: [{scales[0]:.3f}, {scales[1]:.3f}...]"
            else:
                scale_str = ""

            # 1. 检查 Target Stop
            if target_loss is not None and val_loss_val <= target_loss:
                best_val_loss = val_loss_val
                best_step = iter_step
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                stop_reason = f"🎯 Target Stop: 验证损失达到目标 {target_loss:.4f} (当前 {val_loss_val:.4f})"
                print(f"Step {iter_step:4d}/{max_iters} | 训练: {loss.item():.4f} | 验证: {val_loss_val:.4f} | 耗时: {t_elapsed:.1f}s | 🎯 达到目标损失！")
                break

            # 2. 检查 Patience Early Stop
            if val_loss_val < best_val_loss - min_delta:
                best_val_loss = val_loss_val
                best_step = iter_step
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                status_str = f"⭐ 刷新最佳验证损失: {best_val_loss:.4f}"
            else:
                patience_counter += 1
                status_str = f"⏳ 耐心等待 ({patience_counter}/{patience})"

            print(f"Step {iter_step:4d}/{max_iters} | 训练: {loss.item():.4f} | 验证: {val_loss_val:.4f} | 耗时: {t_elapsed:.1f}s | {status_str}{scale_str}")

            if patience_counter >= patience:
                stop_reason = f"🛑 Patience Early Stop: 验证损失连续 {patience} 次未改善 (历史最佳: {best_val_loss:.4f} @ Step {best_step})"
                print(f"\n{stop_reason} -> 自动刹车保护，防止过拟合！")
                break

    print(f"\n[训练终止状态]: {stop_reason}")

    # 恢复最佳权重
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"✨ 已自动加载 Step {best_step} 的历史最佳模型权重 (Val Loss: {best_val_loss:.4f})")

    # 5. 训练完成后的生成测试
    print("\n" + "=" * 80)
    print("【训练后 (偏振门最佳模型) 文本生成实测】:")
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
    print(f"✅ 最优模型检查点已保存至: {ckpt_path}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1: TinyShakespeare Optical NanoGPT")
    parser.add_argument("--attn-type", type=str, default="optical", choices=["optical", "softmax"], help="Attention type")
    parser.add_argument("--pure-optical", action="store_true", default=True, help="Use pure optical mode (no LayerNorm, no sum division, sin^4 MLP)")
    parser.add_argument("--with-layernorm", dest="pure_optical", action="store_false", help="Enable electronic LayerNorm and beam sum normalization")
    parser.add_argument("--iters", type=int, default=1000, help="Max training iterations")
    parser.add_argument("--eval-interval", type=int, default=100, help="Evaluation interval")
    parser.add_argument("--patience", type=int, default=4, help="Patience (number of evals without improvement)")
    parser.add_argument("--min-delta", type=float, default=0.005, help="Minimum delta improvement")
    parser.add_argument("--target-loss", type=float, default=2.45, help="Target validation loss to stop early (default: 2.45)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    train_shakespeare(
        attn_type=args.attn_type,
        pure_optical=args.pure_optical,
        max_iters=args.iters,
        eval_interval=args.eval_interval,
        patience=args.patience,
        min_delta=args.min_delta,
        target_loss=args.target_loss,
        device=args.device
    )
