"""
Step 3: 光电转换断点 —— 模拟现实中光学计算的“光电瓶颈” (O/E-DSP-E/O Bottleneck)
================================================================================
背景:
  光计算在线性矩阵乘法 (GEMM) 具有光速级低延迟与近乎零的动态功耗优势。
  但在非线性层 (如 Softmax / GELU / ReLU) 上，光子之间极难发生相互作用（缺乏天然低功率非线性介质）。
  目前业界的方案必须做“光电混合”：
    [光域线性层 1] 
       --> 【光电探测 O/E + ADC】(强度探测，丢失全部相位！产生皮焦级功耗与纳秒延迟)
       --> 【数字电芯片 (DSP/GPU)】(计算 Softmax)
       --> 【数模转换 DAC + 电光调制 E/O】(强行将实数重编码为光波振幅/相位，产生数分贝插入损耗)
       --> [光域线性层 2]

本脚本直观演示这一断点过程中的三大约束：
  1. 相位坍塌 (Information Loss / Phase Erasure)
  2. 光功率衰减 (Optical Power Loss in dB)
  3. 能耗与延迟的倒挂 (Energy & Latency Disparity)
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

def generate_random_unitary(n=4, seed=42):
    """
    生成一个 n x n 的随机哈尔分布复数酉矩阵 (Haar-random unitary matrix)
    模拟一个由 MZI 阵列（如 Clements 或 Reck 拓扑）构成的无损光子线性网络。
    """
    rng = np.random.default_rng(seed)
    # 高斯随机复数矩阵
    z = rng.standard_normal((n, n)) + 1.0j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q * ph

def softmax(x):
    """标准数值稳定版实数 Softmax"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def run_optoelectronic_bottleneck():
    print("=" * 80)
    print("      ⚡ 光学计算实验 Step 3: 加个“光电转换”断点 (模拟现实瓶颈) ⚡")
    print("=" * 80)
    
    n_dim = 4
    U1 = generate_random_unitary(n=n_dim, seed=101)
    U2 = generate_random_unitary(n=n_dim, seed=202)

    # 1. 初始输入光场 (4个端口，各有特定的振幅与相位)
    print("\n" + "─" * 80)
    print("【阶段 1: 纯光域传播 (Pure Optical Domain - Layer 1)】")
    print("─" * 80)
    # 输入光：具有丰富的相位信息 (Coherent Optical Field)
    amplitudes = np.array([1.0, 0.7, 0.5, 0.8])
    phases = np.array([0.0, np.pi/4, -np.pi/3, np.pi/2])
    E_in = amplitudes * np.exp(1.0j * phases)

    print("输入端口复振幅 E_in:")
    for i in range(n_dim):
        print(f"  Channel {i}: 模长={np.abs(E_in[i]):.3f}, 相位={np.angle(E_in[i]):+6.3f} rad ({np.degrees(np.angle(E_in[i])):5.1f}°) | 表达式={E_in[i]:.3f}")

    total_power_in = np.sum(np.abs(E_in)**2)
    print(f"-> 初始光场总功率 P_in = {total_power_in:.4f} W")

    # 光域线性层计算 (光子穿过芯片，无延迟，无功耗)
    # 模拟波导损耗: 假设芯片长度 1cm, 损耗 0.5 dB/cm
    waveguide_loss_dB = 0.5
    waveguide_trans = 10 ** (-waveguide_loss_dB / 10.0)
    E1 = np.sqrt(waveguide_trans) * (U1 @ E_in)

    print(f"\n光穿过 MZI 阵列完成 U1 @ E_in (耗时 ~35 ps，动态能耗 ~0.05 fJ/MAC):")
    for i in range(n_dim):
        print(f"  Port {i}: 模长={np.abs(E1[i]):.3f}, 相位={np.angle(E1[i]):+6.3f} rad | E1 = {E1[i]:.3f}")
    
    power_after_L1 = np.sum(np.abs(E1)**2)
    print(f"-> Layer 1 输出总光功率 = {power_after_L1:.4f} W (由于波导微损耗，损耗 {10*np.log10(total_power_in/power_after_L1):.2f} dB)")

    # 2. 光电转换断点 (O/E 断点)
    print("\n" + "─" * 80)
    print("【阶段 2: 光电转换断点 (O/E Detection & ADC)】")
    print("─" * 80)
    print("💥 致命时刻 1: 光电二极管 (Photodetector) 是一个“平方律检波器”！")
    print("   它只能感知光强 (光子击中 PN 结激发的电子数 I = |E|^2)，光波所有相位全部坍缩消失！")

    # 光电探测器响应率与插入损耗
    pd_coupling_loss_dB = 1.5  # 探测器耦合损耗
    pd_responsivity = 0.85      # A/W (量子效率)
    pd_power_factor = 10 ** (-pd_coupling_loss_dB / 10.0)

    # 探测到的光强 (实数！)
    I_detected = pd_responsivity * pd_power_factor * (np.abs(E1) ** 2)

    # ADC 模数转换与量化 (模拟 8-bit 量化噪声)
    adc_bits = 8
    max_val = np.max(I_detected) * 1.05
    step = max_val / (2**adc_bits - 1)
    I_adc = np.round(I_detected / step) * step

    print(f"光探测器探测到的电流/强度 I (实数信号，8-bit 量化):")
    for i in range(n_dim):
        lost_phase = np.angle(E1[i])
        print(f"  Channel {i}: I_meas = {I_adc[i]:.4f} | 💀 原本存在的真实物理相位 {lost_phase:+6.3f} rad 被永久丢弃！")

    # 3. 电子非线性计算 (Softmax)
    print("\n" + "─" * 80)
    print("【阶段 3: 传统数字芯片计算 Softmax (Electronic Non-Linearity)】")
    print("─" * 80)
    softmax_out = softmax(I_adc * 5.0)  # 乘以缩放系数以模拟温度/Logits
    print(f"Softmax 概率输出 (实数标量，总和为 1.0):")
    for i in range(n_dim):
        print(f"  Channel {i}: P = {softmax_out[i]:.4f} ({softmax_out[i]*100:5.1f}%)")

    # 4. 电光调制重编码 (E/O 断点)
    print("\n" + "─" * 80)
    print("【阶段 4: 电光调制重编码 (DAC & E/O Modulation)】")
    print("─" * 80)
    print("💥 致命时刻 2: 我们必须拿一束全新的连续激光 (CW Laser)，强行把实数数据重新打回光波！")
    print("   - 方案: 振幅调制 (E = sqrt(P)) 或 相位调制 (Phase = pi * P)")
    print("   - 但原来优雅连贯的波动干涉图样 (Phase Coherence) 已经彻底断代，无法挽回！")
    print("   - 电光调制器 (MZM) 自身存在巨大的插入损耗 (通常 3.5 ~ 6.0 dB) 以及高昂的射频驱动电压功耗！")

    mzm_insertion_loss_dB = 4.0  # 调制器插入损耗 4 dB (只剩下 ~40% 的光能)
    mzm_trans = 10 ** (-mzm_insertion_loss_dB / 10.0)

    # 采用典型的振幅调制重新编码光场:
    # 新的光场振幅与 Softmax 输出成正比，但相位只能人为赋默认值 (比如 0.0)
    E_remodulated = np.sqrt(mzm_trans) * np.sqrt(softmax_out) * np.exp(1.0j * 0.0)

    print(f"\n强行重编码后的复振幅 E2 (调制器损耗 {mzm_insertion_loss_dB} dB):")
    for i in range(n_dim):
        print(f"  Channel {i}: 新模长={np.abs(E_remodulated[i]):.3f}, 强行置零相位={np.angle(E_remodulated[i]):.1f} rad | E = {E_remodulated[i]:.3f}")

    # 5. 第二层光域线性计算
    E_final = U2 @ E_remodulated
    power_final = np.sum(np.abs(E_final)**2)

    # 6. 综合审计报告：光电转换到底让我们付出了多大代价？
    print("\n" + "=" * 80)
    print("        📊 光电转换断点 (O/E-E/O) 系统瓶颈综合审计报告 📊")
    print("=" * 80)

    print(f"{'指标维度':<16} | {'纯光域理想运算 (All-Optical)':<28} | {'光电混合实际运算 (O/E + E/O)'}")
    print("-" * 80)
    print(f"{'延迟 (Latency)':<16} | {'~30 - 50 ps (光速级飞越)':<28} | {'~2 - 5 ns (慢 100 倍! ADC/DAC占90%)'}")
    print(f"{'能耗 (Energy)':<16} | {'< 0.1 fJ / MAC (仅微波导传输)':<28} | {'~10 - 30 pJ / MAC (高 1000 倍! ADC耗电)'}")
    print(f"{'光功率损失':<16} | {'~0.5 dB (仅基础波导衰减)':<28} | {'~6.0 - 10.0 dB (光能衰减 80%~90%)'}")
    print(f"{'相位信息 (Phase)':<16} | {'完美保持相干叠加干涉':<28} | {'在 O/E 处彻底坍缩丢失，人工强行伪造'}")
    print(f"{'光路级联能力':<16} | {'可理论级联多层':<28} | {'需外置光放大器 (SOA/EDFA) 否则信号消失'}")
    print("-" * 80)
    print("💡 结论与启示:")
    print("   这就是为什么当今硅光芯片领域大家常说：")
    print("   『光做矩阵乘法天下第一，但一遇到激活函数/注意力归一化，就全被电接口拖垮了！』")
    print("   这也是目前科研界拼命攻关『全光非线性 (All-Optical Nonlinearity)』的核心动力。")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_optoelectronic_bottleneck()
