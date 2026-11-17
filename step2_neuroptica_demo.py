"""
Step 2 进阶: 使用 Stanford neuroptica 库模拟 Clements 拓扑 MZI 网络
===================================================================
neuroptica (由 Stanford 团队开发) 是专门用于光子神经网络 (ONN) 仿真的开源库。
本脚本演示如何使用 neuroptica 构建多层 MZI 网络，并观察光波在干涉网格中的传播。
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import numpy as np
    import neuroptica as opt
    print("✅ 成功导入 neuroptica 库!")
except ImportError as e:
    print(f"⚠️ 提示: neuroptica 尚未安装完成 ({e})，请稍候再试。")
    sys.exit(0)

def run_neuroptica_demo():
    print("=" * 76)
    print("       🔬 Stanford neuroptica 库: Clements 拓扑 MZI 干涉网格仿真 🔬")
    print("=" * 76)

    # 1. 创建一个 4x4 的 Clements 光学干涉网格
    # Clements 架构使用交错的 2x2 MZI 网格实现任意 N x N 的酉变换 (Unitary Transformation)
    n_ports = 4
    mesh = opt.ClementsLayer(n_ports)

    # 2. 检查网格中包含的 MZI 单元数
    # 对于 N=4，任意酉矩阵需要 N*(N-1)/2 = 6 个 MZI 单元
    print(f"网格端口数: {n_ports} x {n_ports}")
    print(f"网格类型: ClementsLayer (最优正方排布结构)")

    # 3. 构造输入电场向量 (例如在端口 0 注入单色光)
    x = np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    print(f"\n输入光场向量 (单端口注入):")
    print(f"  x = {x}")
    print(f"  输入光强分布: {np.abs(x)**2}")

    # 4. 执行前向光波传播 (Forward Propagation)
    # 在 neuroptica 中，forward_pass 将光波通过各个移相器与分束器传递
    y = mesh.forward_pass(x)

    print(f"\n通过 Clements 网格后的输出复振幅:")
    print(f"  y = {y}")
    
    out_intensities = np.abs(y) ** 2
    print(f"\n各端口输出光强 (平方模长):")
    for i, p in enumerate(out_intensities):
        print(f"  Port {i}: 光强={p:.4f} ({p*100:5.1f}%)")
    
    print(f"\n总输出功率: {np.sum(out_intensities):.4f} (理论应严格等于 1.0，体现能量守恒)")
    print("=" * 76)

if __name__ == "__main__":
    run_neuroptica_demo()
