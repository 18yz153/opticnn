"""
Step 3: Optoelectronic Bottleneck Simulation (O/E - DSP - E/O)
==============================================================
Background:
  Optical computing offers light-speed low latency and near-zero dynamic energy
  for linear matrix multiplications (GEMM).
  However, optical nonlinearities (Softmax, GELU, ReLU) are difficult at low power.
  Current industry systems use optoelectronic hybrid pipelines:
    [Optical Linear Layer 1]
       --> [Optical Detection (O/E) + ADC] (Intensity detected; phase lost! pJ energy, ns delay)
       --> [Digital Chip (DSP/GPU)] (Computes Softmax)
       --> [DAC + Electro-Optic Modulator (E/O)] (Recodes real numbers into light; insertion loss)
       --> [Optical Linear Layer 2]

This script demonstrates three real physical bottlenecks:
  1. Phase loss (information loss)
  2. Optical power loss (insertion loss in dB)
  3. Energy and latency overhead
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

def generate_random_unitary(n=4, seed=42):
    """
    Generate an n x n Haar-random complex unitary matrix.
    Simulates a lossless MZI mesh (Clements or Reck topology).
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, n)) + 1.0j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q * ph

def softmax(x):
    """Standard numerically stable real Softmax."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def run_optoelectronic_bottleneck():
    print("=" * 80)
    print("   Optical Computing Step 3: Optoelectronic Bottleneck (O/E - E/O) Simulation")
    print("=" * 80)

    n_dim = 4
    U1 = generate_random_unitary(n=n_dim, seed=101)
    U2 = generate_random_unitary(n=n_dim, seed=202)

    # 1. Initial input optical field (4 ports with amplitude and phase)
    print("\n" + "-" * 80)
    print("[Stage 1: Pure Optical Domain (Layer 1)]")
    print("-" * 80)
    amplitudes = np.array([1.0, 0.7, 0.5, 0.8])
    phases = np.array([0.0, np.pi/4, -np.pi/3, np.pi/2])
    E_in = amplitudes * np.exp(1.0j * phases)

    print("Input port complex amplitude E_in:")
    for i in range(n_dim):
        print(f"  Channel {i}: Magnitude={np.abs(E_in[i]):.3f}, Phase={np.angle(E_in[i]):+6.3f} rad ({np.degrees(np.angle(E_in[i])):5.1f} deg) | E_in = {E_in[i]:.3f}")

    total_power_in = np.sum(np.abs(E_in)**2)
    print(f"-> Total input optical power P_in = {total_power_in:.4f} W")

    # Optical linear layer propagation
    # Assume 1cm chip, loss 0.5 dB/cm
    waveguide_loss_dB = 0.5
    waveguide_trans = 10 ** (-waveguide_loss_dB / 10.0)
    E1 = np.sqrt(waveguide_trans) * (U1 @ E_in)

    print(f"\nLight passes through MZI mesh U1 @ E_in (~35 ps delay, ~0.05 fJ/MAC energy):")
    for i in range(n_dim):
        print(f"  Port {i}: Magnitude={np.abs(E1[i]):.3f}, Phase={np.angle(E1[i]):+6.3f} rad | E1 = {E1[i]:.3f}")

    power_after_L1 = np.sum(np.abs(E1)**2)
    print(f"-> Layer 1 output power = {power_after_L1:.4f} W (Waveguide loss: {10*np.log10(total_power_in/power_after_L1):.2f} dB)")

    # 2. Optical-to-Electrical conversion (O/E)
    print("\n" + "-" * 80)
    print("[Stage 2: Optical-to-Electrical Conversion (O/E Detection & ADC)]")
    print("-" * 80)
    print("Physical fact 1: Photodetector is a square-law detector.")
    print("It measures power I = |E|^2; phase is completely lost.")

    pd_coupling_loss_dB = 1.5
    pd_responsivity = 0.85      # A/W
    pd_power_factor = 10 ** (-pd_coupling_loss_dB / 10.0)

    # Detected optical power (real scalar)
    I_detected = pd_responsivity * pd_power_factor * (np.abs(E1) ** 2)

    # 8-bit ADC quantization
    adc_bits = 8
    max_val = np.max(I_detected) * 1.05
    step = max_val / (2**adc_bits - 1)
    I_adc = np.round(I_detected / step) * step

    print("Photodetector measured current/intensity I (real value, 8-bit quantized):")
    for i in range(n_dim):
        lost_phase = np.angle(E1[i])
        print(f"  Channel {i}: I_meas = {I_adc[i]:.4f} | Original physical phase {lost_phase:+6.3f} rad discarded!")

    # 3. Electronic computation (Softmax)
    print("\n" + "-" * 80)
    print("[Stage 3: Digital Chip Softmax Computation (Electronic Non-Linearity)]")
    print("-" * 80)
    softmax_out = softmax(I_adc * 5.0)
    print("Softmax probability output (real scalars, sum to 1.0):")
    for i in range(n_dim):
        print(f"  Channel {i}: P = {softmax_out[i]:.4f} ({softmax_out[i]*100:5.1f}%)")

    # 4. Electrical-to-Optical modulation (E/O)
    print("\n" + "-" * 80)
    print("[Stage 4: Electrical-to-Optical Re-modulation (DAC & E/O Modulation)]")
    print("-" * 80)
    print("Physical fact 2: We must use a laser to re-encode numbers into light.")
    print("- Insertion loss of electro-optic modulator (MZM) is typically 3.5 to 6.0 dB.")
    print("- Phase coherence from previous layers is permanently lost.")

    mzm_insertion_loss_dB = 4.0
    mzm_trans = 10 ** (-mzm_insertion_loss_dB / 10.0)

    # Amplitude modulation with dummy phase
    E_remodulated = np.sqrt(mzm_trans) * np.sqrt(softmax_out) * np.exp(1.0j * 0.0)

    print(f"\nRe-modulated complex field E2 (Modulator loss: {mzm_insertion_loss_dB} dB):")
    for i in range(n_dim):
        print(f"  Channel {i}: Magnitude={np.abs(E_remodulated[i]):.3f}, Phase={np.angle(E_remodulated[i]):.1f} rad | E = {E_remodulated[i]:.3f}")

    # 5. Layer 2 optical linear calculation
    E_final = U2 @ E_remodulated
    power_final = np.sum(np.abs(E_final)**2)

    # 6. Audit report
    print("\n" + "=" * 80)
    print("          Optoelectronic Bottleneck (O/E - E/O) Audit Report")
    print("=" * 80)
    print(f"{'Metric':<18} | {'Pure Optical Domain':<28} | {'Hybrid Optoelectronic (O/E + E/O)'}")
    print("-" * 80)
    print(f"{'Latency':<18} | {'~30 - 50 ps (light speed)':<28} | {'~2 - 5 ns (100x slower due to ADC/DAC)'}")
    print(f"{'Energy':<18} | {'< 0.1 fJ / MAC (waveguide)':<28} | {'~10 - 30 pJ / MAC (1000x higher due to ADC)'}")
    print(f"{'Power Loss':<18} | {'~0.5 dB (waveguide attenuation)':<28} | {'~6.0 - 10.0 dB (80% - 90% lost)'}")
    print(f"{'Phase Coherence':<18} | {'Maintained':<28} | {'Lost at O/E, artificially reset at E/O'}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_optoelectronic_bottleneck()
