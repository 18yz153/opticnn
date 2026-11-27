"""
Stage 1: Character-level Shakespeare (TinyShakespeare) Optical NanoGPT Experiment
==================================================================================
Goal:
  Verify gradient propagation and learning capability of optical native attention
  (sin^4 polarization gate) on language modeling tasks.
Criteria:
  - Initial output is random text (Loss ~ 4.18).
  - After training, Loss steadily decreases, and the model generates readable text.
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
    print(f"   🎭 Stage 1: Character-level Shakespeare Optical NanoGPT (Attention: {attn_type.upper()}) 🎭")
    print(f"   Hardware Device: {device.upper()}")
    print(f"   Stop Strategy: MaxIters={max_iters} | Patience={patience} | MinDelta={min_delta} | TargetLoss={target_loss}")
    print("=" * 80)

    # 1. Prepare data
    download_data()
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Dataset size: {len(text):,} chars | Vocab size: {vocab_size}")

    # Simple char tokenizer
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # Train/val split
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

    # 2. Initialize Optical NanoGPT model
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
        attention_type=attn_type,
        normalize_beam=not pure_optical,           # If pure_optical=True, disable dynamic sum division
        mlp_type='optical' if pure_optical else 'gelu', # If pure_optical=True, use optical polarization MLP
        norm_type='none' if pure_optical else 'layernorm' # If pure_optical=True, disable LayerNorm
    )
    audit_optical_pipeline(config)
    model = OpticalNanoGPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e3:.1f} K")
    print(f"Attention type: {attn_type.upper()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # 3. Initial generation before training
    print("\n" + "─" * 80)
    print("[Initial Untrained Text Generation]:")
    start_prompt = "ROMEO:\n"
    context = torch.tensor([encode(start_prompt)], dtype=torch.long, device=device)
    raw_gen = decode(model.generate(context, max_new_tokens=80, temperature=0.8)[0].tolist())
    print(raw_gen)
    print("─" * 80 + "\n")

    # 4. Training loop (Max Iters + Patience Early Stop + Target Stop)
    t0 = time.time()
    best_val_loss = float('inf')
    best_step = 0
    patience_counter = 0
    best_state = None
    stop_reason = "Max iterations reached"

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

            # Inspect optical scale if using optical attention
            if attn_type == 'optical':
                scales = [b.attn.optical_scale.item() for b in model.transformer.h]
                scale_str = f" | Optical Scales: [{scales[0]:.3f}, {scales[1]:.3f}...]"
            else:
                scale_str = ""

            # 1. Check Target Stop
            if target_loss is not None and val_loss_val <= target_loss:
                best_val_loss = val_loss_val
                best_step = iter_step
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                stop_reason = f"Target Stop: Val loss reached {val_loss_val:.4f} <= target {target_loss:.4f}"
                print(f"Step {iter_step:4d}/{max_iters} | Train: {loss.item():.4f} | Val: {val_loss_val:.4f} | Time: {t_elapsed:.1f}s | Target Loss Reached!")
                break

            # 2. Check Patience Early Stop
            if val_loss_val < best_val_loss - min_delta:
                best_val_loss = val_loss_val
                best_step = iter_step
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                status_str = f"New best val loss: {best_val_loss:.4f}"
            else:
                patience_counter += 1
                status_str = f"Patience waiting ({patience_counter}/{patience})"

            print(f"Step {iter_step:4d}/{max_iters} | Train: {loss.item():.4f} | Val: {val_loss_val:.4f} | Time: {t_elapsed:.1f}s | {status_str}{scale_str}")

            if patience_counter >= patience:
                stop_reason = f"Patience Early Stop: Val loss did not improve for {patience} checks (Best: {best_val_loss:.4f} @ Step {best_step})"
                print(f"\n{stop_reason} -> Early stopping to prevent overfitting.")
                break

    print(f"\n[Training Stop Status]: {stop_reason}")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"Loaded best model weights from Step {best_step} (Val Loss: {best_val_loss:.4f})")

    # 5. Generation test after training
    print("\n" + "=" * 80)
    print("[Text Generation with Best Model]:")
    print("=" * 80)
    model.eval()
    with torch.no_grad():
        context = torch.tensor([encode("ROMEO:\n")], dtype=torch.long, device=device)
        out_tokens = model.generate(context, max_new_tokens=250, temperature=0.75, top_k=10)
        generated_text = decode(out_tokens[0].tolist())
        print(generated_text)
    print("=" * 80)

    # Save checkpoint
    ckpt_path = f"optical_shakespeare_{attn_type}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Best model checkpoint saved to: {ckpt_path}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1: TinyShakespeare Optical NanoGPT")
    parser.add_argument("--attn-type", type=str, default="optical", choices=["optical", "softmax", "sa", "soa"], help="Attention type")
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
