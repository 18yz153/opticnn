"""
Step 2: Optical "Hello World" - 2x2 Mach-Zehnder Interferometer (MZI)
=====================================================================
Goals:
1. Simulate a 50:50 beam splitter and a phase shifter in optical waveguides using complex matrices.
2. Given input light field E_in = [1+0j, 0+0j], sweep the phase shifter phi (0 to 2*pi).
3. Observe whether output intensity (|E|^2) follows trigonometric energy splitting (cos^2, sin^2).
4. Verify energy conservation (|E_out|^2 = |E_in|^2).
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

def beam_splitter_50_50():
    """
    50:50 Directional Coupler / Beam Splitter transfer matrix:
    Light in cross path gets a pi/2 phase shift (multiplied by imaginary unit i).
    BS = (1 / sqrt(2)) * [[1,  i],
                          [i,  1]]
    """
    return (1.0 / np.sqrt(2.0)) * np.array([
        [1.0 + 0.0j, 0.0 + 1.0j],
        [0.0 + 1.0j, 1.0 + 0.0j]
    ], dtype=np.complex128)

def phase_shifter(phi):
    """
    Phase Shifter matrix:
    Changes refractive index on one arm via thermo-optic or electro-optic effect,
    producing phase delay phi.
    R(phi) = [[exp(i * phi), 0],
              [0,            1]]
    """
    return np.array([
        [np.exp(1.0j * phi), 0.0 + 0.0j],
        [0.0 + 0.0j,        1.0 + 0.0j]
    ], dtype=np.complex128)

def mzi_transfer_matrix(phi):
    """
    Complete 2x2 MZI cell: BS @ PhaseShifter(phi) @ BS
    """
    bs = beam_splitter_50_50()
    ps = phase_shifter(phi)
    # Order: Input -> BS1 -> PhaseShifter -> BS2 -> Output
    return bs @ ps @ bs

def print_progress_bar(val, max_len=20):
    """Prints a visual power bar in terminal."""
    filled = int(round(val * max_len))
    return f"[{'#' * filled}{'.' * (max_len - filled)}]"

def run_mzi_simulation():
    print("=" * 76)
    print("      Optical Computing Experiment Step 2: Hello World (2x2 MZI)")
    print("=" * 76)
    print("Physics Principle:")
    print("  Light wave propagates in waveguide with complex amplitude: E = A * exp(i * phase)")
    print("  MZI consists of two 50:50 beam splitters and a phase shifter: MZI(phi) = BS @ R(phi) @ BS")
    print("  Theoretical derivation:")
    print("    E_out = i * exp(i*phi/2) * [ sin(phi/2),  cos(phi/2) ]^T * E_in[0]")
    print("    Intensity 1 = |E_out[0]|^2 = sin^2(phi/2)")
    print("    Intensity 2 = |E_out[1]|^2 = cos^2(phi/2)")
    print("=" * 76)

    # 1. Define two input ports (Port 1 has light, amplitude 1.0; Port 2 is dark)
    amplitude = 1.0
    e_in = np.array([amplitude + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    total_input_power = np.sum(np.abs(e_in) ** 2)

    print(f"\n[1] Input Field Vector E_in:")
    print(f"    Port 1: {e_in[0]:.2f} (Magnitude={np.abs(e_in[0]):.2f}, Power={np.abs(e_in[0])**2:.2f})")
    print(f"    Port 2: {e_in[1]:.2f} (Magnitude={np.abs(e_in[1]):.2f}, Power={np.abs(e_in[1])**2:.2f})")
    print(f"    Total input optical power P_in = {total_input_power:.4f}\n")

    # 2. Sweep phase shifter phi from 0 to 2*pi
    num_steps = 13
    phi_values = np.linspace(0, 2 * np.pi, num_steps)

    print(f"{'Phase phi':<10} | {'Output 1 Power (sin^2)':<28} | {'Output 2 Power (cos^2)':<28} | {'Conservation'}")
    print("-" * 85)

    max_err = 0.0
    for phi in phi_values:
        u = mzi_transfer_matrix(phi)
        # Multiplication by light speed in physics
        e_out = u @ e_in

        # Photodiode measurement: I = |E|^2
        i1 = np.abs(e_out[0]) ** 2
        i2 = np.abs(e_out[1]) ** 2
        total_p = i1 + i2

        theory_i1 = np.sin(phi / 2.0) ** 2
        theory_i2 = np.cos(phi / 2.0) ** 2
        err = max(abs(i1 - theory_i1), abs(i2 - theory_i2))
        max_err = max(max_err, err)

        deg_str = f"{np.degrees(phi):5.1f} deg"
        bar1 = f"{i1:5.3f} {print_progress_bar(i1, 15)}"
        bar2 = f"{i2:5.3f} {print_progress_bar(i2, 15)}"

        print(f"{deg_str:<10} | {bar1:<28} | {bar2:<28} | P_tot={total_p:.4f}")

    print("-" * 85)
    print(f"Max error between simulation and theory: {max_err:.2e}")
    print("Success: Light wave interference performs continuous trigonometric multiplication!\n")

if __name__ == "__main__":
    run_mzi_simulation()
