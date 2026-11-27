# OptiCNN: Optical Neural Networks and Optical NanoGPT Simulation

> Exploring optical computing and all-optical Transformers using Python and PyTorch.  
> Supports **NVIDIA CUDA GPU hardware acceleration** and **physical fidelity auditing**.

---

## 📁 Project Structure

```text
d:\opticnn
├── 01_physics_foundations/              # Stage 1: Optical physics first principles
│   ├── step2_mzi_hello_world.py        # Step 2: 2x2 MZI interference and matrix multiplication
│   ├── step2_mzi_plot.py               # Step 2 Visualization: Plot interference curves (mzi_interference.png)
│   ├── step2_neuroptica_demo.py        # Clements topology mesh demo using neuroptica
│   └── step3_optoelectronic_bottleneck.py # Step 3: O/E -> Softmax -> E/O conversion loss audit
│
├── 02_polarization_gates/               # Stage 2: Polarization gates and optical native networks
│   ├── optics_engine.py                # Core engine with 4 physical properties (wavelength, intensity, phase, polarization)
│   ├── experiment_polarization_softmax.py # Physical tests (WDM, Fresnel-Arago, polarization Softmax)
│   └── step4_optical_native_network.py # Native optical activation (sin^4/sin^2) spiral classification (99.4% accuracy)
│
├── 03_optical_nanogpt/                  # Stage 3: All-optical Native NanoGPT language model
│   ├── model.py                        # Optical CausalSelfAttention (sin^4 gate, SA, SOA) & physical audit
│   ├── train_shakespeare.py            # Stage 1: Character-level Shakespeare language modeling
│   ├── train_toy_math.py               # Stage 2: Toy math addition (data routing and winner-take-all)
│   └── data/
│       ├── prepare_shakespeare.py      # TinyShakespeare dataset downloader and split
│       └── prepare_toy_math.py         # Addition dataset generator
│
├── .gitignore                          # Git ignore configuration
├── requirements.txt                    # Python dependencies (with PyTorch CUDA)
└── README.md                           # This document
```

---

## 🔬 Physical Fidelity Audit: Optics vs Electronics

When studying optical computing, we must clearly separate what happens optically from what uses digital electronics:

| Step in Pipeline | Formula | Physical Carrier | Implementation Type | Engineering Reality |
| :--- | :--- | :--- | :--- | :--- |
| **Q, K, V Projections** | $x W_q, x W_k, x W_v$ | Clements / Reck silicon MZI mesh | 🟢 **Pure Optical (picoseconds)** | Passive coherent interference; light passing through waveguides computes unitary matrix multiplication |
| **Attention Dot Product** | $Q K^T$ | Free-space diffraction / waveguide mesh | 🟢 **Pure Optical (speed of light)** | Coherent superposition naturally computes the inner product |
| **Causal Attention Mask** | `masked_fill(0.0)` | Spatial aperture / opaque absorber | 🟢 **Pure Optical (physical blocking)** | Transmission is exactly 0; light is physically absorbed or reflected |
| **Attention Activation** | $\sin^4(\theta)$ | Birefringent Lyot polarization filter | 🟢 **Pure Optical (Lyot filter)** | Squared Malus law transmission; continuous and super-linear |
| **Value Aggregation** | $\text{Att} \cdot V$ | Multi-waveguide interference | 🟢 **Pure Optical (passive sum)** | Photons add linearly with zero delay and zero energy |
| **Residual Connection** | $x + \text{Attn}(x)$ | 3dB directional coupler | 🟢 **Pure Optical (passive device)** | Passive interference combines two optical beams |
| **Attention Normalization** (Passive) | $A \times \frac{1}{\sqrt{T}}$ | 1×T passive beam splitter | 🟢 **Pure Optical (zero computation)** | Natural beam splitting attenuation without any division operation |
| **Attention Normalization** (Dynamic) | $\frac{A}{\sum A}$ | Digital floating-point ALU | 🔴 **Digital Electronics (bottleneck)** | Requires measuring total beam power and computing division on a digital chip |
| **LayerNorm** | $\frac{x - \mu}{\sigma} \gamma + \beta$ | Digital DSP mean/variance unit | 🔴 **Digital Electronics (bottleneck)** | Calculating global mean and variance requires digital registers; removed in pure optical mode |
| **MLP Activation** | $\text{GELU}(z), \text{ReLU}(z)$ | Electronic ALU | 🔴 **Digital Electronics (bottleneck)** | Replaced with learnable $\sin^4$ polarization gates or saturable absorbers in optical mode |
| **Token Embedding** | $\text{Embedding}(idx)$ | Laser diode array + E/O modulator | 🟡 **Optoelectronic Boundary (E/O)** | Modulates discrete token IDs onto laser intensity and phase |
| **Output Decoding** | $\text{Argmax}(\text{detectors})$ | Photodetector array (PD) + ADC readout | 🟡 **Optoelectronic Boundary (O/E)** | Converts light intensity to current; picks the highest current channel as the predicted token |

---

## ⚡ CUDA Acceleration (RTX 4070 Laptop GPU)

With `torch==2.6.0+cu124`, training runs on an **NVIDIA GeForce RTX 4070 Laptop GPU**:
* **Speedup**: From 78.9 seconds on CPU down to **9.8 seconds** on GPU (~8x faster).
* **VRAM**: Low memory footprint (< 500 MB), fast forward passes and gradient updates.

---

## 🚀 Running Experiments

Activate your virtual environment and run:

### 1. Stage 1: Character-level Shakespeare (TinyShakespeare)
```bash
# Pure optical mode (no LayerNorm, passive 1/sqrt(T) attenuation, optical sin^4 MLP)
python 03_optical_nanogpt/train_shakespeare.py --pure-optical --iters 1000 --eval-interval 100 --patience 4 --device cuda

# Hybrid mode (with LayerNorm and beam sum normalization)
python 03_optical_nanogpt/train_shakespeare.py --with-layernorm --iters 1000 --device cuda
```
* **Performance**: In pure optical mode, 800 iterations complete in ~9.8s; Loss drops from 4.18 to 2.51.
* **Text Output**: Learns word boundaries, spacing, and play dialogue format (`ROMEO:\nTh t t wrd is s ble winouthar thine...`).
* **Conclusion**: **Shows that $\sin^4(\theta)$ polarization gates can backpropagate gradients for language modeling through optical layers.**

### 2. Stage 2: Toy Math Addition (Data Routing)
```bash
# Hybrid mode with Automatic Gain Control (AGC / beam sum normalization)
python 03_optical_nanogpt/train_toy_math.py --iters 1500 --eval-interval 150 --patience 4 --target-acc 85.0 --device cuda

# Pure passive mode (no electronic normalization; tests passive beam attenuation)
python 03_optical_nanogpt/train_toy_math.py --pure-optical --iters 1200 --device cuda
```
* **Performance**:
  * **Hybrid AGC mode**: Reaches **76% to 85%** accuracy on unseen test addition problems (`7+7=14`, `2+5=07`, `16+22=038` are predicted correctly).
  * **Pure passive mode**: Without normalization or gain, power decays across layers. This highlights the practical need for **Semiconductor Optical Amplifiers (SOA)** or **Saturable Absorbers (SA)** in deep optical chips.
* **Conclusion**: **Super-linear $\sin^4(\theta)$ response routes attention to target number tokens (winner-take-all behavior).**

---

## 🛑 Three-Tier Stopping Strategy

To avoid unnecessary training steps and prevent overfitting:
1. **Target Stop**:
   * **Math task**: Stops when test accuracy reaches `target_acc` (e.g. 85.0%).
   * **Language task**: Stops when validation loss reaches `target_loss`.
2. **Patience Early Stop**:
   * Monitors validation loss. If there is no improvement over `patience` evaluation checks, training stops automatically.
   * **Automatic Best Weights Restore**: The best historical checkpoint is restored when training ends.
3. **Max Iterations**:
   * Fixed upper bound on total iterations to prevent infinite loops.
