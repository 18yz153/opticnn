"""
Step 2: 跑通光的 "Hello World" —— 2x2 Mach-Zehnder 干涉仪 (MZI)
=============================================================
目标:
1. 用复数 (Complex Matrix) 模拟物理光波导中的 50:50 分束器 (Beam Splitter) 与移相器 (Phase Shifter)。
2. 给定输入光场 E_in = [1+0j, 0+0j]，调节移相器相位 phi (0 到 2*pi)。
3. 观测输出光强 (模平方 |E|^2) 是否严格服从三角函数能量分配 (cos^2, sin^2)。
4. 验证能量守恒 (酉矩阵保范性 |E_out|^2 = |E_in|^2)。
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

def beam_splitter_50_50():
    """
    50:50 定向耦合分束器 (Directional Coupler / Beam Splitter) 传输矩阵:
    光在交叉耦合路径上会积累 pi/2 (即乘上虚数 i) 的相移。
    BS = (1 / sqrt(2)) * [[1,  i],
                          [i,  1]]
    """
    return (1.0 / np.sqrt(2.0)) * np.array([
        [1.0 + 0.0j, 0.0 + 1.0j],
        [0.0 + 1.0j, 1.0 + 0.0j]
    ], dtype=np.complex128)

def phase_shifter(phi):
    """
    移相器 (Phase Shifter) 矩阵:
    在其中一条光波导臂上通过热光效应 (微加热器) 或电光效应改变折射率，产生相位延迟 phi。
    R(phi) = [[exp(i * phi), 0],
              [0,            1]]
    """
    return np.array([
        [np.exp(1.0j * phi), 0.0 + 0.0j],
        [0.0 + 0.0j,        1.0 + 0.0j]
    ], dtype=np.complex128)

def mzi_transfer_matrix(phi):
    """
    一个完整的 2x2 MZI 单元 = BS1 @ PhaseShifter(phi) @ BS2
    """
    bs = beam_splitter_50_50()
    ps = phase_shifter(phi)
    # 光波依次经过: 输入 -> BS1 -> 移相器 -> BS2 -> 输出
    # 矩阵乘法顺序: BS @ PS @ BS @ E_in
    return bs @ ps @ bs

def print_progress_bar(val, max_len=20):
    """用于在终端打印直观的光强能量条"""
    filled = int(round(val * max_len))
    return f"[{'#' * filled}{'.' * (max_len - filled)}]"

def run_mzi_simulation():
    print("=" * 76)
    print("           ✨ 光学计算实验 Step 2: 跑通光的 Hello World (2x2 MZI) ✨")
    print("=" * 76)
    print("物理原理:")
    print("  光波在硅光波导中以复振幅传播: E = A * exp(i * phase)")
    print("  MZI 由两组 50:50 分束器和中间的移相器组成: MZI(phi) = BS @ R(phi) @ BS")
    print("  理论推导:")
    print("    E_out = i * exp(i*phi/2) * [ sin(phi/2),  cos(phi/2) ]^T * E_in[0]")
    print("    Intensity 1 = |E_out[0]|^2 = sin^2(phi/2)")
    print("    Intensity 2 = |E_out[1]|^2 = cos^2(phi/2)")
    print("=" * 76)

    # 1. 定义两束输入光 (Port 1 有光，振幅为 1.0；Port 2 为暗通道)
    amplitude = 1.0
    e_in = np.array([amplitude + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    total_input_power = np.sum(np.abs(e_in) ** 2)

    print(f"\n[1] 输入光场向量 E_in:")
    print(f"    Port 1: {e_in[0]:.2f} (复振幅模长={np.abs(e_in[0]):.2f}, 功率={np.abs(e_in[0])**2:.2f})")
    print(f"    Port 2: {e_in[1]:.2f} (复振幅模长={np.abs(e_in[1]):.2f}, 功率={np.abs(e_in[1])**2:.2f})")
    print(f"    总输入光功率 P_in = {total_input_power:.4f}\n")

    # 2. 扫描移相器的相位 phi (0 到 2*pi)
    num_steps = 13
    phi_values = np.linspace(0, 2 * np.pi, num_steps)

    print(f"{'相位 phi':<10} | {'输出1强度 (sin^2)':<28} | {'输出2强度 (cos^2)':<28} | {'总能量守恒'}")
    print("-" * 85)

    max_err = 0.0
    for phi in phi_values:
        # 获取当前相位的 MZI 传输矩阵
        u = mzi_transfer_matrix(phi)

        # 光速计算：矩阵乘法（在真实芯片中，光线穿过芯片即完成计算，延迟仅皮秒级！）
        e_out = u @ e_in

        # 光探测器 (Photodiode) 测量光强度: I = |E|^2
        i1 = np.abs(e_out[0]) ** 2
        i2 = np.abs(e_out[1]) ** 2
        total_p = i1 + i2

        # 理论值对比
        theory_i1 = np.sin(phi / 2.0) ** 2
        theory_i2 = np.cos(phi / 2.0) ** 2
        err = max(abs(i1 - theory_i1), abs(i2 - theory_i2))
        max_err = max(max_err, err)

        deg_str = f"{np.degrees(phi):5.1f} deg"
        bar1 = f"{i1:5.3f} {print_progress_bar(i1, 15)}"
        bar2 = f"{i2:5.3f} {print_progress_bar(i2, 15)}"

        print(f"{deg_str:<10} | {bar1:<28} | {bar2:<28} | P_tot={total_p:.4f}")

    print("-" * 85)
    print(f"✅ 数值模拟与解析解最大误差: {max_err:.2e}")
    print("🎉 恭喜！你成功用电磁波干涉原理完成了连续的三角函数权重分配乘法运算！")
    print("   这不仅是数学上的向量旋转，更是光波在空间中以光速发生的天然线性代数计算。\n")

if __name__ == "__main__":
    run_mzi_simulation()
