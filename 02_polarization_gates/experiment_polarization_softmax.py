"""
Experiment: Exploring Optical Softmax with Polarization Gates
============================================================
Tests and demonstrates:
  1. Wavelength Division Multiplexing (WDM): Multiple wavelengths in a single waveguide without crosstalk.
  2. Polarization and interference coupling (Fresnel-Arago laws): Only parallel components interfere.
  3. Polarization gate for Softmax:
     - Polarization Beam Splitter (PBS) naturally conserves energy: cos^2(theta) + sin^2(theta) = 1.0.
     - Cascade birefringent filter (Lyot filter) provides sin^4(theta) steep non-linearity.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from optics_engine import OpticalField, LaserEmitter, OpticalReceiver, PolarizationGate

def demo_wavelength_multiplexing():
    print("\n" + "=" * 76)
    print("[Experiment 1: Wavelength Division Multiplexing (WDM)]")
    print("=" * 76)
    print("Principle: Different optical wavelengths do not interfere and share the same waveguide.")

    laser_1550 = LaserEmitter(wavelength_nm=1550.0)
    laser_1310 = LaserEmitter(wavelength_nm=1310.0)

    f1 = laser_1550.emit(intensity=1.5, phase_deg=30.0, polarization_angle_deg=0.0)
    f2 = laser_1310.emit(intensity=2.5, phase_deg=90.0, polarization_angle_deg=45.0)

    print(f"Signal 1 -> {f1}")
    print(f"Signal 2 -> {f2}")

    receiver = OpticalReceiver()
    result = receiver.combine_and_detect([f1, f2])

    print("\nOptical Receiver (Spectrometer Demodulation):")
    for ch, p in result["channels"].items():
        print(f"  Channel {ch:<8}: Power = {p:.3f} W (zero crosstalk)")
    print(f"  Total power in waveguide = {result['total_power']:.3f} W")
    print("Conclusion: WDM enables parallel multi-channel computing in a single waveguide.")

def demo_polarization_interference_coupling():
    print("\n" + "=" * 76)
    print("[Experiment 2: Polarization and Interference (Fresnel-Arago Laws)]")
    print("=" * 76)
    print("Principle: Only parallel polarization states interfere; orthogonal states never interfere.")

    laser = LaserEmitter(wavelength_nm=1550.0)
    receiver = OpticalReceiver()

    # Case A: Parallel polarization (H), 180 deg phase difference (destructive interference)
    f_h1 = laser.emit(intensity=1.0, phase_deg=0.0, polarization_angle_deg=0.0)
    f_h2 = laser.emit(intensity=1.0, phase_deg=180.0, polarization_angle_deg=0.0)
    res_a = receiver.combine_and_detect([f_h1, f_h2])

    # Case B: Orthogonal polarization (H and V), 180 deg phase difference
    f_h = laser.emit(intensity=1.0, phase_deg=0.0, polarization_angle_deg=0.0)
    f_v = laser.emit(intensity=1.0, phase_deg=180.0, polarization_angle_deg=90.0)
    res_b = receiver.combine_and_detect([f_h, f_v])

    print(f"Case A (Parallel, opposite phase): Detected power = {res_a['total_power']:.4f} W (Destructive interference!)")
    print(f"Case B (Orthogonal, opposite phase): Detected power = {res_b['total_power']:.4f} W (Cannot interfere, scalar sum!)")
    print("Conclusion: Polarization acts as an on/off gate for optical interference.")

def demo_polarization_softmax():
    print("\n" + "=" * 76)
    print("[Experiment 3: Polarization Gate for Softmax]")
    print("=" * 76)
    print("Goal: Given input logits, simulate winner-take-all and sum-to-1 normalization.")

    def standard_softmax(x, T=1.0):
        e_x = np.exp((x - np.max(x)) / T)
        return e_x / np.sum(e_x)

    def polarization_gate_softmax(logits, k=0.85):
        # Electro-optic rotation angle (0 to pi/2)
        mean_val = np.mean(logits)
        sigmoid_response = 1.0 / (1.0 + np.exp(-k * (logits - mean_val)))
        angles_rad = (np.pi / 2.0) * sigmoid_response

        # Two-stage Lyot filter cascade transmission
        cascade_transmission = np.sin(angles_rad) ** 4

        # PBS global normalization
        return cascade_transmission / np.sum(cascade_transmission)

    test_cases = [
        np.array([2.0, 1.0, 0.1]),
        np.array([0.5, 0.5, 0.5]),
        np.array([4.0, 0.2, -1.0]),
    ]

    for idx, logits in enumerate(test_cases, 1):
        p_ideal = standard_softmax(logits, T=1.0)
        p_polar = polarization_gate_softmax(logits, k=0.85)

        print(f"\n[Test Case {idx}] Input Logits: {logits}")
        print(f"{'Channel':<8} | {'Standard Softmax':<20} | {'Polarization Gate':<25} | {'Error'}")
        print("-" * 65)
        for ch in range(len(logits)):
            err = abs(p_ideal[ch] - p_polar[ch])
            print(f"Ch {ch:<5} | {p_ideal[ch]:8.4f} ({p_ideal[ch]*100:5.1f}%) | "
                  f"{p_polar[ch]:8.4f} ({p_polar[ch]*100:5.1f}%)       | {err:6.4f}")

        print(f"Sum check: Standard sum={np.sum(p_ideal):.4f}, Polar sum={np.sum(p_polar):.4f}")

    print("\n" + "=" * 76)
    print("Key Conclusions:")
    print("1. PBS splits light into orthogonal components, where cos^2(theta) + sin^2(theta) = 1.")
    print("2. Cascaded Lyot waveplates produce sin^4(theta) steepness, mimicking exp(x) winner-take-all.")
    print("=" * 76 + "\n")

if __name__ == "__main__":
    demo_wavelength_multiplexing()
    demo_polarization_interference_coupling()
    demo_polarization_softmax()
