# Optical Computing Hardware Catalog: Building Blocks for Photonic Transformers

> **Purpose**: A comprehensive reference of physically realizable optical and optoelectronic building blocks.  
> This catalog details the physical mechanism, mathematical transfer function, size, energy cost, and fabrication feasibility for each unit, mapping each directly to Transformer operations ($QKV$ projections, Attention routing, Normalization, MLP activations, and Memory).

---

## 🗺️ Master Comparison Matrix

| Optical Unit | Hardware Mechanism | Math Operation $y = f(x)$ | Footprint / Size | Energy / Power | Maturity & Fab Cost | Transformer Mapping |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. MZI Mesh** | Mach-Zehnder Interferometer (Phase Shifters + 3dB Couplers) | Unitary Matrix Multiplication $U(\theta, \phi) \mathbf{x}$ | $50\text{--}200\,\mu\text{m}$ per unit cell | Thermo: $10\text{--}20\,\text{mW}$<br>EO: $<0.1\,\text{pJ/bit}$ | Mature (Silicon SOI PDK); Low cost | $W_Q, W_K, W_V, W_O$ projections, linear MLP |
| **2. MRR Weight Bank** | Microring Resonator array coupled to bus waveguide | WDM Weighted Sum $y_j = \sum_i w_{ji} x_i$ | $\approx 10 \times 10\,\mu\text{m}$ ($r \approx 3\text{--}5\,\mu\text{m}$) | $\sim 1\text{--}5\,\text{mW}$ (thermal tuning) | Mature (SOI); Low cost, requires thermal trim | Dense Matrix-Vector Multiplication, compact cross-attention |
| **3. 3dB Coupler / Y-Junction** | Directional coupler / Multi-Mode Interference (MMI) | Split / Combine $\frac{1}{\sqrt{2}}(x_1 + x_2)$ | $10\text{--}30\,\mu\text{m} \times 2\text{--}5\,\mu\text{m}$ | **0 mW (100% passive)** | Ultra-mature; Negligible cost | Residual addition ($x + \text{Attn}$), beam splitting |
| **4. Diffractive Surface (D2NN)** | Metasurface / Lithographic phase mask arrays | Free-Space Fourier / Convolution $\iint E \frac{e^{ikr}}{r} dS$ | $0.5\text{--}2\,\text{mm}$ (stacked 3D) | **0 mW (100% passive)** | Mature lithography ($\text{Si}_3\text{N}_4 / \text{TiO}_2$); Moderate cost | $Q K^T$ dot product over full context in parallel |
| **5. Lyot Polarization Gate** | Birefringent crystal / waveplate cascade | Super-linear Malus response $\sin^4(\theta)$ | $50\text{--}100\,\mu\text{m}$ | **0 mW passive** (sub-pJ if modulated) | Moderate (TFLN or waveplate integration) | Softmax numerator replacement, optical MLP activation |
| **6. Saturable Absorber (SA)** | 2D Material (Graphene / $MoS_2$) Pauli blocking | Intensity-dependent transmission $T_0 + \frac{\Delta T \cdot I^2}{I_s^2 + I^2}$ | $10\text{--}30\,\mu\text{m} \times 0.5\,\mu\text{m}$ | **0 mW (Passive optical thresholding)** | Emerging (CVD transfer on SOI); Low material cost | Sigmoid/ReLU replacement, dead-zone prevention |
| **7. Optical Amplifier (SOA)** | III-V Microcavity stimulated emission / carrier depletion | Cross-Gain Modulation $G = \frac{G_0}{1 + \sum P_j / P_s}$ | $200\text{--}500\,\mu\text{m} \times 10\,\mu\text{m}$ | **High: 50–100 mA bias (~150 mW)** | Low yield (InP heterogeneous bonding); Expensive | **Physical Softmax division** (winner-take-all without ALU) |
| **8. Phase-Change Material (PCM)** | Non-volatile GST / GSST phase transition | Non-volatile weight transmission $T(\text{cryst/amorp})$ | $1\text{--}5\,\mu\text{m}$ patch | **0 static power**; Switching: $\sim 1\text{--}10\,\text{pJ}$ | Emerging (backend integration); Moderate cost | Non-volatile weight storage ($W$), optical KV-Cache |
| **9. Electro-Absorption Modulator** | QCSE / Franz-Keldysh absorption in micro-junction | Voltage-to-intensity modulation $T(V) = e^{-\alpha(V) L}$ | $10\text{--}50\,\mu\text{m}$ | Ultra-low: $< 10\,\text{fJ/bit}$ | Mature (standard in telecom); Low-to-moderate | Token embedding injection (E/O boundary) |
| **10. Spiral Delay Line** | Low-loss coiled waveguide delay buffer | Time delay $y(t) = x(t - \Delta t)$, $\Delta t = \frac{n_g L}{c}$ | $100 \times 100\,\mu\text{m}$ (for ~1 ns delay) | **0 mW (100% passive)** | Ultra-mature (SOI / $\text{Si}_3\text{N}_4$); Very cheap | Autoregressive token alignment, Recurrent/Mamba state buffer |
| **11. Analog AGC Feedback** | 1% Photodiode tap + micro CMOS op-amp + VOA | Dynamic normalization $y = x / (\alpha \sum x_i)$ | $50 \times 50\,\mu\text{m}$ | Very low: $0.2\text{--}0.5\,\text{mW}$ | Very mature (Standard CMOS co-packaged) | Softmax denominator, LayerNorm variance stabilizer |

---

## 🔍 Deep Dive by Functional Category

### 1. Linear Operators & Matrix Projections (Q, K, V Projections, Feed-Forward Projections)

#### A. Mach-Zehnder Interferometer (MZI) Meshes
* **Physical Principle**: A beam splitter divides light into two arms; phase shifters ($\Delta \theta, \Delta \phi$) introduce relative phase shifts before a second beam splitter causes constructive/destructive interference.
* **Math Transfer Function**:
  $$U(\theta, \phi) = \begin{bmatrix} e^{i\phi}\cos(\theta/2) & -e^{i\phi}\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{bmatrix}$$
* **Arbitrary Matrix Synthesis**: According to the **Clements Theorem**, an $N \times N$ mesh of $\frac{N(N-1)}{2}$ MZIs can synthesize **any arbitrary $N \times N$ unitary matrix** $U$. Using Singular Value Decomposition (SVD):
  $$W = U \cdot \Sigma \cdot V^\dagger$$
  Any arbitrary weight matrix $W$ is mapped to two MZI unitary meshes ($U$ and $V^\dagger$) separated by optical attenuators/amplifiers ($\Sigma$).
* **Trade-offs**: Extremely high fidelity, but scaling to large dimensions (e.g., $d=4096$) requires thousands of MZIs and large die area.

#### B. Microring Resonator (MRR) Weight Banks
* **Physical Principle**: Optical waveguides are evanescently coupled to micro-rings. When the wavelength of light matches the resonant frequency of the ring, light couples into the ring and drops. By tuning the resonance (thermally or electro-optically), light transmission is modulated continuously between 0 and 1.
* **Math Transfer Function**:
  $$T_i(\lambda_i) = \frac{(1 - r_i^2)^2}{(1 - r_i^2)^2 + 4 r_i^2 \sin^2(\pi n_{eff} L / \lambda_i)}$$
* **Wavelength Division Multiplexing (WDM)**: Inputs on $K$ distinct wavelengths ($\lambda_1, \dots, \lambda_K$) propagate in a single waveguide. A bank of $K$ rings weighs each channel. A balanced photodiode sums all intensities:
  $$I_{out} = \sum_{k=1}^K w_k I_{in}(\lambda_k)$$
* **Trade-offs**: 20x to 50x smaller than an MZI, but sensitive to ambient temperature drift; requires active thermal tracking.

---

### 2. Attention Scoring & Free-Space Token Mixing

#### A. Diffractive Optical Neural Networks (D2NN / Metasurfaces)
* **Physical Principle**: Light propagates through free space or low-loss dielectric layers. Micro-patterned meta-atoms impart local phase and amplitude shifts across the continuous 2D wavefront.
* **Math Transfer Function**: Rayleigh-Sommerfeld diffraction:
  $$E(x,y,z) = \frac{z}{i\lambda} \iint E(x',y',0) \frac{e^{ikR}}{R^2} dx' dy', \quad R = \sqrt{(x-x')^2 + (y-y')^2 + z^2}$$
* **Transformer Mapping**: Can perform massive parallel $O(1)$ matrix multiplications across all tokens simultaneously in free space with zero dynamic energy.
* **Trade-offs**: Fixed passive weights (difficult to dynamically reconfigure at runtime unless coupled with liquid crystal or phase-change spatial light modulators).

---

### 3. All-Optical Nonlinearity & Activation (Softmax Numerator, MLP Activation)

#### A. Birefringent Lyot Polarization Gate
* **Physical Principle**: An optical field with polarization angle $\theta$ passes through birefringent crystal plates placed between cross-polarizers. Phase retardations create a quartic interference envelope.
* **Math Transfer Function**:
  $$T(\theta) = \sin^4(\theta)$$
* **Transformer Mapping**: Super-linear response acts as an effective winner-take-all function for attention weights and serves as an optical activation function in MLP layers.
* **Trade-offs**: Gradient vanishes near zero ($4\theta^3$ dead zone), which requires optical bias or saturable absorbers to avoid parameter freezing.

#### B. 2D Material Saturable Absorbers (SA)
* **Physical Principle**: Monolayer graphene or transition metal dichalcogenides ($MoS_2$, $WS_2$) possess finite electron states. At high optical intensities, the conduction band states fill up, and photons can no longer be absorbed due to **Pauli blocking**, making the material transparent.
* **Math Transfer Function**:
  $$T(I) = T_0 + (1 - T_0)\frac{(I / I_{sat})^p}{1 + (I / I_{sat})^p}, \quad p \in \{1, 2\}$$
* **Transformer Mapping**: Ideal optical substitute for Sigmoid or ReLU. Fully passive, zero electrical power, and provides a continuous non-zero derivative across low intensities.
* **Trade-offs**: Requires evanescent waveguide coupling over tens of microns to achieve deep modulation contrast.

---

### 4. Normalization & Physical Division (Softmax Denominator & LayerNorm)

#### A. Semiconductor Optical Amplifier (SOA) Cross-Gain Modulation (XGM)
* **Physical Principle**: When multiple optical beams enter an active III-V semiconductor microcavity, stimulated emission consumes injected carrier density. The shared carrier pool creates natural competition: stronger channels deplete the gain available to all channels.
* **Math Transfer Function**:
  $$G = \frac{G_0}{1 + \sum_{j=1}^M \frac{P_j}{P_{sat}}}$$
* **Transformer Mapping**: **Computes relative division physically without any digital ALUs**. Naturally implements the partition function of Softmax.
* **Trade-offs**: High static current consumption (50–100 mA), thermal dissipation, and requires heterogeneous InP/Si bonding.

#### B. Micro-Analog AGC (Automatic Gain Control) Feedback
* **Physical Principle**: A 1% passive directional coupler taps optical power into a compact photodiode. A miniature analog circuit regulates an integrated Variable Optical Attenuator (VOA) via negative feedback.
* **Math Transfer Function**:
  $$P_{out} = \frac{P_{in}}{1 + K \cdot P_{tap}} \approx \frac{P_{in}}{\alpha \sum P_j + \epsilon}$$
* **Transformer Mapping**: Replicates LayerNorm and Softmax denominator normalization with minimal area ($< 50 \times 50\,\mu\text{m}$) and sub-milliwatt power.
* **Trade-offs**: Introduces analog circuit boundary, though power and delay are orders of magnitude lower than digital DSPs.

---

### 5. Memory, Token Buffers & Non-Volatile Weight Storage

#### A. Phase-Change Materials (PCM, e.g. GST / GSST)
* **Physical Principle**: Nanoscale patches of chalcogenide glass switched between amorphous (low index, high transmission) and crystalline (high index, high absorption) states via laser pulses.
* **Math Transfer Function**: Non-volatile optical weighting with continuous analog levels:
  $$w \in [w_{min}, w_{max}], \quad \text{retains state for }>10\text{ years without power}$$
* **Transformer Mapping**: Zero-static-power storage for weights and optical KV-Cache.
* **Trade-offs**: Finite write cycle endurance ($10^6\text{--}10^8$ switches), though read operations are completely non-destructive and instantaneous.

#### B. Waveguide Optical Delay Lines
* **Physical Principle**: Coiled low-loss silicon or silicon nitride ($\text{Si}_3\text{N}_4$) waveguides introduce physical propagation time delays.
* **Math Transfer Function**:
  $$E_{out}(t) = E_{in}(t - \Delta t), \quad \Delta t = \frac{n_g L}{c}$$
* **Transformer Mapping**: Provides time alignment for auto-regressive tokens and implements state-space memory buffers (e.g., in recurrent or Mamba architectures).

---

## 💡 Practical Transformer Assembly Guide

When assembling a realistic Photonic Transformer, hardware units can be mapped as follows:

```
┌────────────────────────────────────────────────────────────────────────┐
│                       OPTICAL TRANSFORMER BLOCK                        │
└────────────────────────────────────────────────────────────────────────┘

 [Input Waveguide Tokens] 
            │
            ▼
 1. Linear Projections (Q, K, V) ──► Clements MZI Meshes or MRR Banks (Passive / EO)
            │
            ▼
 2. Attention Scoring (Q @ K^T)  ──► Spatial Waveguide Crossbar or Diffractive Metasurface
            │
            ▼
 3. Attention Activation         ──► Birefringent Lyot Gate (sin^4) + 2D Saturable Absorber (SA)
            │
            ▼
 4. Softmax Normalization        ──► SOA Carrier Depletion (All-Optical) OR Analog AGC VOA Loop
            │
            ▼
 5. Value Summation (Attn @ V)   ──► Passive Multi-Mode Interference (MMI) Waveguide Combiners
            │
            ▼
 6. Residual Addition            ──► 3dB Directional Couplers (100% Passive)
            │
            ▼
 7. Layer Normalization          ──► Analog AGC Feedback Circuit (Rescales variance)
            │
            ▼
 8. Feed-Forward MLP             ──► SVD-decomposed MZI Mesh + SA/Polarization Nonlinearity
            │
            ▼
 [Next Layer Waveguide]
```
