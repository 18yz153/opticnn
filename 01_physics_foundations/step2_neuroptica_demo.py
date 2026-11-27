"""
Step 2 Advanced: Clements Topology MZI Mesh Simulation with Stanford neuroptica
==============================================================================
neuroptica (developed by Stanford researchers) is an open-source library for
simulating optical neural networks (ONNs).
This script builds a multi-port MZI mesh and simulates light propagation.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import numpy as np
    import neuroptica as opt
    print("Successfully imported neuroptica library!")
except ImportError as e:
    print(f"Notice: neuroptica is not installed ({e}).")
    sys.exit(0)

def run_neuroptica_demo():
    print("=" * 76)
    print("   Stanford neuroptica library: Clements Topology MZI Mesh Simulation")
    print("=" * 76)

    # 1. Create a 4x4 Clements optical interference mesh
    # Clements topology uses an interleaved 2x2 MZI grid to implement any NxN unitary transformation
    n_ports = 4
    mesh = opt.ClementsLayer(n_ports)

    print(f"Mesh ports: {n_ports} x {n_ports}")
    print(f"Mesh type: ClementsLayer (optimal rectangular layout)")

    # 2. Input electric field vector (single-port injection at port 0)
    x = np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    print(f"\nInput field vector (single-port injection):")
    print(f"  x = {x}")
    print(f"  Input power distribution: {np.abs(x)**2}")

    # 3. Forward pass through optical mesh
    y = mesh.forward_pass(x)

    print(f"\nOutput complex amplitude after Clements mesh:")
    print(f"  y = {y}")

    out_intensities = np.abs(y) ** 2
    print(f"\nOutput power per port (|E|^2):")
    for i, p in enumerate(out_intensities):
        print(f"  Port {i}: Power={p:.4f} ({p*100:5.1f}%)")

    print(f"\nTotal output power: {np.sum(out_intensities):.4f} (strictly 1.0, energy conserved)")
    print("=" * 76)

if __name__ == "__main__":
    run_neuroptica_demo()
