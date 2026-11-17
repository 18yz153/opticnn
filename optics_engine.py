"""
OptiCNN: 全自由度物理光场仿真引擎 (optics_engine.py)
=====================================================
建模光的 4 大物理属性:
  1. 波长 / 频率 (Wavelength / Frequency) -> 实现 WDM 波分复用合成
  2. 振幅 / 强度 (Amplitude / Intensity)  -> 携带能量与信息
  3. 相位 / 干涉 (Phase / Interference)   -> 空间相干叠加
  4. 偏振态 (Polarization / Jones Vector) -> 空间电场矢量方向 [Ex, Ey]^T
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

class OpticalField:
    """
    单个单色光场基元 (Monochromatic Optical Field Mode)
    """
    def __init__(self, wavelength_nm=1550.0, amplitude=1.0, phase_rad=0.0, jones_vector=None):
        self.wavelength_nm = float(wavelength_nm)  # 波长 (默认电信 C 波段 1550 nm)
        self.amplitude = float(amplitude)          # 电场幅度 (实数 >= 0)
        self.phase_rad = float(phase_rad % (2 * np.pi)) # 相位 (0 到 2*pi)
        
        # 偏振态: 归一化复数琼斯向量 [Ex, Ey]^T
        # [1, 0] 为水平偏振 (H), [0, 1] 为垂直偏振 (V)
        # [1, 1]/sqrt(2) 为 45° 对角偏振, [1, -i]/sqrt(2) 为右旋圆偏振
        if jones_vector is None:
            self.jones = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
        else:
            j = np.array(jones_vector, dtype=np.complex128)
            norm = np.linalg.norm(j)
            self.jones = j / norm if norm > 1e-12 else np.array([1.0+0j, 0+0j])

    @property
    def intensity(self):
        """光强度 (光子通量 / 模平方) I = |A|^2"""
        return self.amplitude ** 2

    @property
    def complex_field(self):
        """完整空间复电场矢量 E = A * exp(i * phase) * Jones"""
        return self.amplitude * np.exp(1.0j * self.phase_rad) * self.jones

    def __repr__(self):
        # 计算偏振角与椭圆率
        h_power = np.abs(self.jones[0])**2
        v_power = np.abs(self.jones[1])**2
        return (f"OpticalField(λ={self.wavelength_nm:.1f}nm, I={self.intensity:.3f}, "
                f"Phase={np.degrees(self.phase_rad):.1f}°, H/V=[{h_power:.2f}, {v_power:.2f}])")


class LaserEmitter:
    """
    激光发射器 (可调节波长、强度、初始相位、偏振态)
    """
    def __init__(self, wavelength_nm=1550.0):
        self.wavelength_nm = wavelength_nm

    def emit(self, intensity=1.0, phase_deg=0.0, polarization_angle_deg=0.0):
        """
        发射一束光:
        polarization_angle_deg: 0° 为水平线偏(H), 90° 为垂直线偏(V), 45° 为对角线偏
        """
        theta = np.radians(polarization_angle_deg)
        jones = np.array([np.cos(theta), np.sin(theta)], dtype=np.complex128)
        return OpticalField(
            wavelength_nm=self.wavelength_nm,
            amplitude=np.sqrt(max(intensity, 0.0)),
            phase_rad=np.radians(phase_deg),
            jones_vector=jones
        )


class OpticalReceiver:
    """
    智能光接收器 (可切换探测模式):
    1. 平方律强度探测 (Square-law Intensity)
    2. 偏振态偏振计分解 (Polarization Resolver: 分离 H 和 V 偏振分量)
    3. 光谱仪分解 (Spectrometer: 区分不同波长的通道)
    """
    def detect_intensity(self, field: OpticalField):
        """探测标量光强 I = |E|^2"""
        return field.intensity

    def detect_polarization_split(self, field: OpticalField):
        """
        通过偏振分束器 (PBS) 探测 H 与 V 偏振各自的光强
        天然满足: I_H + I_V = I_total
        """
        total_I = field.intensity
        I_h = total_I * (np.abs(field.jones[0]) ** 2)
        I_v = total_I * (np.abs(field.jones[1]) ** 2)
        return {"I_H": I_h, "I_V": I_v, "Total": total_I}

    def combine_and_detect(self, fields: list[OpticalField]):
        """
        合成多个输入光波并进行相干/非相干探测:
        - 相同波长 + 相同偏振 -> 相干干涉叠加 (振幅相加)
        - 不同波长 或 正交偏振 -> 非相干能量累加 (光强直接相加)
        """
        # 按波长分组
        wl_groups = {}
        for f in fields:
            wl = round(f.wavelength_nm, 2)
            wl_groups.setdefault(wl, []).append(f)

        total_measured_power = 0.0
        details = {}

        for wl, grp in wl_groups.items():
            # 相同波长下，矢量复振幅相干求和
            total_E = np.zeros(2, dtype=np.complex128)
            for f in grp:
                total_E += f.complex_field
            
            p_wl = np.sum(np.abs(total_E) ** 2)
            details[f"{wl}nm"] = p_wl
            total_measured_power += p_wl

        return {"total_power": total_measured_power, "channels": details}


class PolarizationGate:
    """
    偏振调制门 (Polarization Gate):
    利用电光效应 (Pockels Cell) 或非线性偏振旋转 (NPR) 改变光场的偏振态
    """
    @staticmethod
    def rotate(field: OpticalField, theta_rad: float):
        """旋转偏振方向 (波片/旋光器)"""
        rot_matrix = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad),  np.cos(theta_rad)]
        ], dtype=np.complex128)
        new_jones = rot_matrix @ field.jones
        return OpticalField(
            wavelength_nm=field.wavelength_nm,
            amplitude=field.amplitude,
            phase_rad=field.phase_rad,
            jones_vector=new_jones
        )

    @staticmethod
    def phase_retard(field: OpticalField, delta_phi_rad: float):
        """给垂直分量引入相对于水平分量的相位延迟 (双折射晶体 / Pockels cell)"""
        retard_matrix = np.array([
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, np.exp(1.0j * delta_phi_rad)]
        ], dtype=np.complex128)
        new_jones = retard_matrix @ field.jones
        return OpticalField(
            wavelength_nm=field.wavelength_nm,
            amplitude=field.amplitude,
            phase_rad=field.phase_rad,
            jones_vector=new_jones
        )
