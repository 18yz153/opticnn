"""
Stage 2: Toy Math Addition Optical NanoGPT Experiment
======================================================
Goal:
  Verify data routing and winner-take-all behavior of optical native attention
  (sin^4 polarization gate) on arithmetic tasks.
Criteria:
  - Input prompt format: "15+32="
  - Model generates: "047\\n"
  - Evaluate accuracy across test addition problems.
"""

import os
import sys
import time
import random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from model import OpticalNanoGPT, GPTConfig, audit_optical_pipeline
from data.prepare_toy_math import generate_math_dataset, FILE_PATH

def quick_evaluate_math(model, device, encode, decode, num_questions=25):
    """Quick evaluation of math addition accuracy for early stopping."""
    model.eval()
    correct = 0
    rng = random.Random(777)
    with torch.no_grad():
        for i in range(num_questions):
            if i % 3 == 0:
                a = rng.randint(10, 45)
                b = rng.randint(10, 45)
                expected = f"{a+b:03d}"
                n_tokens = 3
            else:
                a = rng.randint(0, 9)
                b = rng.randint(0, 9)
                expected = f"{a+b:02d}"
                n_tokens = 2
            prompt = f"{a}+{b}="
            ctx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
            out = model.generate(ctx, max_new_tokens=n_tokens, temperature=0.01)
            pred = decode(out[0].tolist())[len(prompt):len(prompt)+n_tokens]
            if pred == expected:
                correct += 1
    model.train()
    return correct / num_questions * 100.0

def train_toy_math(attn_type='optical', pure_optical=False, max_iters=1500,
                   eval_interval=150, patience=4, min_delta=0.005,
                   target_acc=85.0, target_loss=None,
                   device='cuda' if torch.cuda.is_available() else 'cpu'):
    print("=" * 80)
    print(f"   🧮 Stage 2: Toy Math Addition Routing Test (Attention: {attn_type.upper()}) 🧮")
    print(f"   Hardware Device: {device.upper()}")
    print(f"   Stop Strategy: MaxIters={max_iters} | Patience={patience} | MinDelta={min_delta} | TargetAcc={target_acc}%")
    print("=" * 80)

    # 1. Prepare data: single-digit and double-digit addition samples
    random.seed(42)
    lines = []
    for _ in range(6000):
        # 75% single-digit addition, 25% double-digit addition
        if random.random() < 0.75:
            a = random.randint(0, 9)
            b = random.randint(0, 9)
            lines.append(f"{a}+{b}={a+b:02d}\n")
        else:
            a = random.randint(10, 50)
            b = random.randint(10, 50)
            lines.append(f"{a}+{b}={a+b:03d}\n")

    chars = sorted(list(set("".join(lines))))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    print(f"Vocab chars: {chars} (Vocab size: {vocab_size})")

    # Split train and test sets
    random.seed(123)
    random.shuffle(lines)
    n_train = int(len(lines) * 0.85)
    train_lines = lines[:n_train]
    test_lines = lines[n_train:]

    train_data = torch.tensor(encode("".join(train_lines)), dtype=torch.long)
    test_data = torch.tensor(encode("".join(test_lines)), dtype=torch.long)

    block_size = 16  # Single line "15+32=047\n" length is 10
    batch_size = 64

    def get_batch(data_tensor):
        ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
        x = torch.stack([data_tensor[i:i + block_size] for i in ix])
        y = torch.stack([data_tensor[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # 2. Initialize Optical NanoGPT
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=3,
        n_head=4,
        n_embd=64,
        attention_type=attn_type,
        normalize_beam=not pure_optical,
        mlp_type='optical' if pure_optical else 'gelu',
        norm_type='none' if pure_optical else 'layernorm'
    )
    audit_optical_pipeline(config)
    model = OpticalNanoGPT(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e3:.1f} K")
    print(f"Attention type: {attn_type.upper()}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)

    # 3. Training loop (Max Iters + Target Stop + Patience Early Stop)
    t0 = time.time()
    best_val_loss = float('inf')
    best_acc = 0.0
    best_step = 0
    patience_counter = 0
    best_state = None
    stop_reason = "Max iterations reached"

    for iter_step in range(1, max_iters + 1):
        xb, yb = get_batch(train_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter_step % eval_interval == 0 or iter_step == max_iters:
            model.eval()
            with torch.no_grad():
                xt, yt = get_batch(test_data)
                _, val_loss = model(xt, yt)
            model.train()
            
            val_loss_val = val_loss.item()
            t_elapsed = time.time() - t0

            # Quick evaluation of math accuracy (25 questions)
            cur_acc = quick_evaluate_math(model, device, encode, decode, num_questions=25)
            scale_info = f" | Optical Scale: {model.transformer.h[0].attn.optical_scale.item():.4f}" if attn_type == 'optical' else ""

            # 1. Check Target Accuracy Stop
            if target_acc is not None and cur_acc >= target_acc:
                best_acc = cur_acc
                best_val_loss = val_loss_val
                best_step = iter_step
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                stop_reason = f"Target Accuracy Stop: Accuracy reached {cur_acc:.1f}% >= target {target_acc:.1f}%"
                print(f"Step {iter_step:4d}/{max_iters} | Train Loss: {loss.item():.4f} | Test Loss: {val_loss_val:.4f} | Acc: {cur_acc:.1f}% | Time: {t_elapsed:.1f}s | Target Accuracy Reached!")
                break

            # 2. Check Target Loss Stop
            if target_loss is not None and val_loss_val <= target_loss:
                best_val_loss = val_loss_val
                best_step = iter_step
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                stop_reason = f"Target Loss Stop: Val loss reached {val_loss_val:.4f} <= target {target_loss:.4f}"
                print(f"Step {iter_step:4d}/{max_iters} | Train Loss: {loss.item():.4f} | Test Loss: {val_loss_val:.4f} | Acc: {cur_acc:.1f}% | Time: {t_elapsed:.1f}s | Target Loss Reached!")
                break

            # 3. Check Patience Early Stop
            if val_loss_val < best_val_loss - min_delta or cur_acc > best_acc:
                if val_loss_val < best_val_loss - min_delta:
                    best_val_loss = val_loss_val
                if cur_acc > best_acc:
                    best_acc = cur_acc
                best_step = iter_step
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                status_str = f"New best (Loss: {best_val_loss:.4f}, Acc: {best_acc:.1f}%)"
            else:
                patience_counter += 1
                status_str = f"Patience waiting ({patience_counter}/{patience})"

            print(f"Step {iter_step:4d}/{max_iters} | Train Loss: {loss.item():.4f} | Test Loss: {val_loss_val:.4f} | Acc: {cur_acc:.1f}% | Time: {t_elapsed:.1f}s | {status_str}{scale_info}")

            if patience_counter >= patience:
                stop_reason = f"Patience Early Stop: No improvement for {patience} checks (Best: Loss={best_val_loss:.4f}, Acc={best_acc:.1f}% @ Step {best_step})"
                print(f"\n{stop_reason} -> Early stopping to prevent overfitting.")
                break

    print(f"\n[Training Stop Status]: {stop_reason}")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"Loaded best model weights from Step {best_step}!")

    # 4. Accurate arithmetic evaluation (50 unseen random problems)
    print("\n" + "=" * 80)
    print("[Test Set Arithmetic Addition Evaluation (50 Random Problems - Best Model)]:")
    print("=" * 80)

    model.eval()
    correct = 0
    total = 50
    sample_display = []

    random.seed(999)
    for i in range(total):
        if i % 3 == 0:
            a = random.randint(10, 45)
            b = random.randint(10, 45)
            expected_ans = f"{a+b:03d}"
            prompt = f"{a}+{b}="
            n_tokens = 3
        else:
            a = random.randint(0, 9)
            b = random.randint(0, 9)
            expected_ans = f"{a+b:02d}"
            prompt = f"{a}+{b}="
            n_tokens = 2

        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            out_tokens = model.generate(context, max_new_tokens=n_tokens, temperature=0.01)
            gen_str = decode(out_tokens[0].tolist())
            pred_ans = gen_str[len(prompt):len(prompt)+n_tokens]

        is_correct = (pred_ans == expected_ans)
        if is_correct:
            correct += 1

        if i < 15: # Print first 15 samples
            mark = "CORRECT" if is_correct else "WRONG"
            sample_display.append(f"  {prompt:<8} Expected={expected_ans} | Pred={pred_ans} | {mark}")

    print("\n".join(sample_display))
    acc = correct / total * 100.0
    print("-" * 80)
    print(f"Final Arithmetic Accuracy: {correct}/{total} ({acc:.1f}%)")
    print("=" * 80)

    if acc >= 90.0:
        print("Conclusion: Optical attention achieves high accuracy on arithmetic token routing.\n")
    else:
        print("Note: Accuracy can be further improved with more iterations or parameter tuning.\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2: Toy Math Addition Optical NanoGPT")
    parser.add_argument("--attn-type", type=str, default="optical", choices=["optical", "softmax", "sa", "soa"], help="Attention type")
    parser.add_argument("--pure-optical", action="store_true", default=False, help="Use pure optical mode (passive 1/sqrt(T), sin^4 MLP, no LayerNorm)")
    parser.add_argument("--iters", type=int, default=1500, help="Max training iterations")
    parser.add_argument("--eval-interval", type=int, default=150, help="Evaluation interval")
    parser.add_argument("--patience", type=int, default=4, help="Patience (number of evals without improvement)")
    parser.add_argument("--min-delta", type=float, default=0.005, help="Minimum delta improvement")
    parser.add_argument("--target-acc", type=float, default=85.0, help="Target test accuracy to stop early (e.g. 85.0 or 95.0)")
    parser.add_argument("--target-loss", type=float, default=None, help="Target validation loss to stop early")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    train_toy_math(
        attn_type=args.attn_type,
        pure_optical=args.pure_optical,
        max_iters=args.iters,
        eval_interval=args.eval_interval,
        patience=args.patience,
        min_delta=args.min_delta,
        target_acc=args.target_acc,
        target_loss=args.target_loss,
        device=args.device
    )
