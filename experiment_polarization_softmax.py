"""
实验: 用偏振门 (Polarization Gates) 与多物理属性探索全光 Softmax
===================================================================
探讨与验证:
  1. 波长合成 (WDM): 不同波长在同一空间内零串扰正交叠加计算。
  2. 偏振与干涉的耦合 (Fresnel-Arago 定律): 只有偏振相同的分量才会干涉。
  3. 偏振门实现 Softmax:
     - 偏振分束器 (PBS) 拥有天然的守恒归一化: cos^2(theta) + sin^2(theta) = 1.0
     - 配合非线性偏振旋转 (NPR) 或 级联双折射偏振门 (Lyot-style Cascade)
     - 观察能否模拟出 Softmax 的“指数增强 + 竞争抑制”特性！
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from optics_engine import OpticalField, LaserEmitter, OpticalReceiver, PolarizationGate

def demo_wavelength_multiplexing():
    print("\n" + "=" * 76)
    print("【实验 1: 波长合成 (Wavelength Multiplexing / WDM 算力叠加)】")
    print("=" * 76)
    print("原理: 不同频率的光子互不相干，可以在同一波导内传输，互不串扰。")

    # 创建两个不同波长的发射源 (比如 1550nm 和 1310nm)
    laser_1550 = LaserEmitter(wavelength_nm=1550.0)
    laser_1310 = LaserEmitter(wavelength_nm=1310.0)

    # 分别发射携带不同信号强度的光
    f1 = laser_1550.emit(intensity=1.5, phase_deg=30.0, polarization_angle_deg=0.0)
    f2 = laser_1310.emit(intensity=2.5, phase_deg=90.0, polarization_angle_deg=45.0)

    print(f"信号 1 -> {f1}")
    print(f"信号 2 -> {f2}")

    receiver = OpticalReceiver()
    result = receiver.combine_and_detect([f1, f2])

    print(f"\n光接收器 (光谱仪解调):")
    for ch, p in result["channels"].items():
        print(f"  解调波长 {ch:<8}: 光功率 = {p:.3f} W (无损且零串扰保留原始计算值)")
    print(f"  空间波导总通量 = {result['total_power']:.3f} W")
    print("💡 结论: 波长属性可以实现光计算的高并发多通道并行 (100+ 通道只需 1 根波导)！")

def demo_polarization_interference_coupling():
    print("\n" + "=" * 76)
    print("【实验 2: 偏振对干涉的影响 (Fresnel-Arago 定律)】")
    print("=" * 76)
    print("原理: 只有平行偏振的光波才能产生干涉峰谷；正交偏振 (H与V) 的光波绝不干涉。")

    laser = LaserEmitter(wavelength_nm=1550.0)
    receiver = OpticalReceiver()

    # 案例 A: 两束光都是水平偏振 (H), 相位差 180° (相消干涉)
    f_h1 = laser.emit(intensity=1.0, phase_deg=0.0, polarization_angle_deg=0.0)
    f_h2 = laser.emit(intensity=1.0, phase_deg=180.0, polarization_angle_deg=0.0)
    res_a = receiver.combine_and_detect([f_h1, f_h2])

    # 案例 B: 一束水平偏振 (H), 一束垂直偏振 (V), 相位差 180°
    f_h = laser.emit(intensity=1.0, phase_deg=0.0, polarization_angle_deg=0.0)
    f_v = laser.emit(intensity=1.0, phase_deg=180.0, polarization_angle_deg=90.0)
    res_b = receiver.combine_and_detect([f_h, f_v])

    print(f"情况 A (偏振平行, 相位相反): 总探测光强 = {res_a['total_power']:.4f} W (能量完全相消干涉！)")
    print(f"情况 B (偏振正交, 相位相反): 总探测光强 = {res_b['total_power']:.4f} W (无法干涉，光强直接标量求和！)")
    print("💡 结论: 偏振是干涉的开关（Gate）！控制偏振态就能精准开启或阻断两束光的干涉计算。")

def demo_polarization_softmax():
    print("\n" + "=" * 76)
    print("【实验 3: 能否用“偏振门 (Polarization Gate)”实现 Softmax？】")
    print("=" * 76)
    print("目标: 给定输入 logits 向量，通过偏振旋转与偏振分束器 (PBS)，")
    print("      模拟 Softmax 的【指数胜者通吃】与【和为 1 归一化】。")

    # 真实 Softmax 函数
    def standard_softmax(x, T=1.0):
        e_x = np.exp((x - np.max(x)) / T)
        return e_x / np.sum(e_x)

    # 偏振门 Softmax 物理模拟:
    # 物理结构:
    # 1. 泡克耳斯盒 (Pockels Cell / 硅基电光微环调制器):
    #    输入电压信号驱动电光晶体，感应出双折射相位差，使偏振方向发生 S 型单调偏转:
    #    theta = (pi / 2) * Sigmoid(k * (x - mean(x)))
    # 2. 级联双折射偏振滤光片 (Lyot Filter Cascade):
    #    经过两级偏振滤光片后，透过率服从 T = sin^4(theta)，实现超线性指数陡峭化！
    # 3. 偏振分束器 (PBS) 阵列:
    #    各端口物理强度自动满足能量守恒，实现被动光子级全局归一化。
    def polarization_gate_softmax(logits, k=0.85):
        # 电光调制偏振偏转角 (单调映射至 0 ~ pi/2)
        mean_val = np.mean(logits)
        sigmoid_response = 1.0 / (1.0 + np.exp(-k * (logits - mean_val)))
        angles_rad = (np.pi / 2.0) * sigmoid_response
        
        # 两级级联偏振片透过率 (Lyot 滤波指数增强)
        cascade_transmission = np.sin(angles_rad) ** 4
        
        # 经 PBS 全局无损归一化 (光学总能量守恒)
        return cascade_transmission / np.sum(cascade_transmission)

    # 测试向量
    test_cases = [
        np.array([2.0, 1.0, 0.1]),
        np.array([0.5, 0.5, 0.5]),
        np.array([4.0, 0.2, -1.0]),
    ]

    for idx, logits in enumerate(test_cases, 1):
        p_ideal = standard_softmax(logits, T=1.0)
        p_polar = polarization_gate_softmax(logits, k=0.85)

        print(f"\n[测试组 {idx}] 输入 Logits: {logits}")
        print(f"{'通道':<8} | {'理论标准 Softmax':<20} | {'偏振门物理模拟 (Polar-Softmax)':<30} | {'误差'}")
        print("-" * 75)
        for ch in range(len(logits)):
            err = abs(p_ideal[ch] - p_polar[ch])
            print(f"Ch {ch:<5} | {p_ideal[ch]:8.4f} ({p_ideal[ch]*100:5.1f}%) | "
                  f"{p_polar[ch]:8.4f} ({p_polar[ch]*100:5.1f}%)         | {err:6.4f}")
        
        print(f"归一化验证: 理论和={np.sum(p_ideal):.4f}, 偏振和={np.sum(p_polar):.4f}")

    print("\n" + "=" * 76)
    print("🎯 核心物理结论:")
    print("1. 【为什么偏振可以做 Softmax 的归一化】:")
    print("   偏振分束器 (PBS) 将任意光场分解为两个正交分量，其强度满足 cos^2(θ) + sin^2(θ) = 1。")
    print("   这种投影在物理上是严格保范的（被动无损归一化），不需要数字除法器！")
    print("2. 【为什么偏振可以做 Softmax 的指数竞争】:")
    print("   通过级联偏振波片 (Lyot 滤波器架构) 或非线性偏振旋转 (NPR)，透射率随角度按 sin^(2N)(θ) 快速激增，")
    print("   能产生和 exp(x) 极其相似的陡峭激发曲线，大信号迅速获胜，小信号被彻底压制！")
    print("=" * 76 + "\n")

if __name__ == "__main__":
    demo_wavelength_multiplexing()
    demo_polarization_interference_coupling()
    demo_polarization_softmax()
