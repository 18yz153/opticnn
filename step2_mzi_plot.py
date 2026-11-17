"""
Step 2 扩展: 绘制 MZI 干涉与能量分配曲线 (保存为 mzi_interference.png)
================================================================
本脚本将 2x2 MZI 的干涉光强与理论三角函数分布绘制成精美图表。
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib.pyplot as plt

def generate_mzi_plot():
    # 50:50 耦合分束器
    bs = (1.0 / np.sqrt(2.0)) * np.array([
        [1.0 + 0.0j, 0.0 + 1.0j],
        [0.0 + 1.0j, 1.0 + 0.0j]
    ], dtype=np.complex128)

    # 输入光场: 单端口注入，振幅为 1.0
    e_in = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

    # 扫描相位 phi (0 到 4*pi)
    phis = np.linspace(0, 4 * np.pi, 500)
    i1_list = []
    i2_list = []
    total_list = []

    for phi in phis:
        ps = np.array([
            [np.exp(1.0j * phi), 0.0 + 0.0j],
            [0.0 + 0.0j,        1.0 + 0.0j]
        ], dtype=np.complex128)
        u = bs @ ps @ bs
        e_out = u @ e_in
        i1 = np.abs(e_out[0]) ** 2
        i2 = np.abs(e_out[1]) ** 2
        i1_list.append(i1)
        i2_list.append(i2)
        total_list.append(i1 + i2)

    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(phis / np.pi, i1_list, label=r'Port 1 Intensity ($I_1 = \sin^2(\phi/2)$)', color='#0066cc', lw=2)
    plt.plot(phis / np.pi, i2_list, label=r'Port 2 Intensity ($I_2 = \cos^2(\phi/2)$)', color='#ff6600', lw=2)
    plt.plot(phis / np.pi, total_list, label=r'Total Optical Power ($I_1 + I_2 = 1.0$)', color='#2ca02c', ls='--', lw=1.5)

    plt.title("2x2 Mach-Zehnder Interferometer (MZI) Transfer Function", fontsize=14, fontweight='bold')
    plt.xlabel(r'Phase Shift $\phi$ ($\times \pi$ rad)', fontsize=12)
    plt.ylabel('Normalized Optical Power / Intensity', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.15)
    plt.legend(fontsize=11, loc='center right')
    
    # 标注干涉极值点
    plt.annotate('Constructive Interference (Port 1)', xy=(1.0, 1.0), xytext=(1.2, 1.05),
                 arrowprops=dict(arrowstyle='->', color='#0066cc', lw=1.5), fontsize=10, color='#0066cc')
    plt.annotate('Constructive Interference (Port 2)', xy=(0.0, 1.0), xytext=(0.2, 0.9),
                 arrowprops=dict(arrowstyle='->', color='#ff6600', lw=1.5), fontsize=10, color='#ff6600')

    output_path = "mzi_interference.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"📊 干涉仿真曲线已成功保存至: {output_path}")

if __name__ == "__main__":
    generate_mzi_plot()
