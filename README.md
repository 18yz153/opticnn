# OptiCNN: Optical Neural Networks and Photonic Transformer Simulation

> Exploring optical computing and all-optical Transformers using Python and PyTorch.  
> Designed with **strict physical fidelity auditing** to separate true optical physics from hidden electronic computation.

---

## 📁 Project Structure

```text
d:\opticnn
├── 01_physics_foundations/              # Stage 1: Optical physics first principles
│   ├── step2_mzi_hello_world.py        # 2x2 MZI interference and matrix multiplication
│   ├── step2_mzi_plot.py               # Visualizing MZI interference curves
│   ├── step2_neuroptica_demo.py        # Clements topology mesh demo using neuroptica
│   └── step3_optoelectronic_bottleneck.py # O/E -> Softmax -> E/O conversion loss audit
│
├── 02_polarization_gates/               # Stage 2: Polarization gates and optical native networks
│   ├── optics_engine.py                # Jones vector engine (wavelength, intensity, phase, polarization)
│   ├── experiment_polarization_softmax.py # WDM, Fresnel-Arago interference, and polarization tests
│   └── step4_optical_native_network.py # Native optical activation (sin^4/sin^2) spiral classification
│
├── 03_optical_nanogpt/                  # Stage 3: Optical Native NanoGPT language model
│   ├── model.py                        # Optical Attention (sin^4, SA, SOA, Softmax) & physical audit
│   ├── train_shakespeare.py            # Stage 1: Character-level Shakespeare language modeling
│   ├── train_toy_math.py               # Stage 2: Toy math addition (data routing and winner-take-all)
│   └── data/
│       ├── prepare_shakespeare.py      # TinyShakespeare dataset downloader and split
│       └── prepare_toy_math.py         # Arithmetic addition dataset generator
│
├── OPTICAL_HARDWARE_CATALOG.md         # Comprehensive catalog of optical hardware units (MZI, SA, SOA, PCM...)
├── EXPERIMENT_REPORT.md                # 3-architecture evaluation (Electronic vs Pure Optical vs Hybrid AGC)
├── REPORT_SA_SOA_VS_ELECTRONIC.md      # In-depth benchmark of SA and SOA vs Electronic Softmax
├── requirements.txt                    # Python dependencies
└── README.md                           # Main documentation
```

---

## 🧩 Optical Hardware Building Blocks

The project models real-world optical components based on experimental physics. See **[OPTICAL_HARDWARE_CATALOG.md](OPTICAL_HARDWARE_CATALOG.md)** for detailed mathematical functions, footprint sizes, power consumption, and fabrication costs:

* **Linear Transformations**: Clements Mach-Zehnder Interferometer (MZI) meshes and Microring Resonator (MRR) banks for $W_Q, W_K, W_V, W_O$ projections.
* **Passive Operations**: 3dB Directional Couplers for residual connections ($x + \text{Attn}$) and $1/\sqrt{T}$ passive beam splitters.
* **Nonlinear Activations**:
  * **Lyot Polarization Gates ($\sin^4\theta$)**: Birefringent crystal filters providing super-linear transmission.
  * **2D Saturable Absorbers (SA)**: Monolayer graphene / $MoS_2$ Pauli-blocking for all-optical thresholding without gradient dead zones.
* **Normalization & Division**:
  * **Semiconductor Optical Amplifiers (SOA)**: Cross-gain modulation (XGM) in optical microcavities providing physical partition-function division.
  * **Analog AGC Feedback Loops**: Miniature photodiode tap + variable optical attenuator (VOA) for power stabilization.
* **Optical Memory & State Buffers**: Phase-change materials (PCM) for non-volatile weights and spiral waveguides for time delays.

---

## 🔬 Physical Fidelity Audit: Optics vs Electronics

When studying optical computing, we must separate operations that occur in light from those requiring digital electronics:

| Pipeline Operation | Formula | Physical Carrier | Implementation Type | Engineering Reality |
| :--- | :--- | :--- | :--- | :--- |
| **Q, K, V Projections** | $x W_q, x W_k, x W_v$ | Clements / Reck silicon MZI mesh | 🟢 **Pure Optical (picoseconds)** | Passive coherent interference; light passing through waveguides computes unitary matrix multiplication |
| **Attention Dot Product** | $Q K^T$ | Free-space diffraction / waveguide mesh | 🟢 **Pure Optical (speed of light)** | Coherent superposition naturally computes inner products |
| **Causal Attention Mask** | `masked_fill(0.0)` | Spatial aperture / opaque absorber | 🟢 **Pure Optical (physical blocking)** | Transmission is exactly 0; light is physically absorbed |
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

## 🛠️ Simulation Setup

Install dependencies in Python 3.10+:

```bash
pip install -r requirements.txt
```

PyTorch automatically uses GPU acceleration when CUDA is available.

---

## 🚀 Running Experiments

The model supports 4 attention mechanisms (`--attn-type`):
* `optical`: Birefringent Lyot polarization gate ($\sin^4\theta$)
* `sa`: 2D material Saturable Absorber (Pauli blocking)
* `soa`: Semiconductor Optical Amplifier (Cross-Gain Modulation)
* `softmax`: Traditional electronic baseline

### 1. Stage 1: Character-level Shakespeare (Language Modeling)

Tests gradient backpropagation and text modeling through multi-layer optical networks:

```bash
# 1. Pure optical mode (sin^4 gate, passive 1/sqrt(T), optical MLP, no LayerNorm)
python 03_optical_nanogpt/train_shakespeare.py --attn-type optical --pure-optical --iters 1000 --eval-interval 100 --patience 4

# 2. Saturable Absorber (SA) mode (smooth gradients, eliminates dead zone)
python 03_optical_nanogpt/train_shakespeare.py --attn-type sa --pure-optical --iters 1000

# 3. Semiconductor Optical Amplifier (SOA) mode
python 03_optical_nanogpt/train_shakespeare.py --attn-type soa --pure-optical --iters 1000

# 4. Optoelectronic Hybrid mode (with electronic LayerNorm and beam sum normalization)
python 03_optical_nanogpt/train_shakespeare.py --attn-type optical --with-layernorm --iters 1000

# 5. Traditional Electronic Softmax baseline
python 03_optical_nanogpt/train_shakespeare.py --attn-type softmax --with-layernorm --iters 1000
```

### 2. Stage 2: Toy Math Addition (Data Routing & Winner-Take-All)

Tests whether the optical attention mechanism can focus sharply on relevant tokens to perform arithmetic addition (e.g., `16+22=038`):

```bash
# 1. Hybrid mode with Automatic Gain Control (AGC) - reaches ~85% accuracy
python 03_optical_nanogpt/train_toy_math.py --attn-type optical --iters 1500 --eval-interval 150 --target-acc 85.0

# 2. SOA Physical Softmax mode (achieves ~68% accuracy with ZERO digital division)
python 03_optical_nanogpt/train_toy_math.py --attn-type soa --pure-optical --iters 1500

# 3. Saturable Absorber (SA) passive mode
python 03_optical_nanogpt/train_toy_math.py --attn-type sa --pure-optical --iters 1500

# 4. Pure passive mode (no gain, tests passive beam attenuation)
python 03_optical_nanogpt/train_toy_math.py --attn-type optical --pure-optical --iters 1200
```

---

## 📊 Benchmark Summary Across Architectures

| Architecture Mode | Shakespeare Val Loss | Toy Math Accuracy | Digital Division (`/ sum`) | Key Observation |
| :--- | :--- | :--- | :--- | :--- |
| **Electronic Softmax Baseline** | **2.26** | **64% ~ 78%** | Yes (Floating-point ALU) | Standard baseline; high accuracy but high power consumption |
| **Polarization Gate ($\sin^4$) Pure Optical** | 2.50 ~ 2.51 | 6.0% | No (Passive $1/\sqrt{T}$) | Words form, but cubic derivative ($4x^3$) causes a gradient dead zone near zero |
| **Polarization Gate + Electronic AGC** | **2.43** | **76% ~ 85%** | Yes (Analog/digital AGC) | Strong winner-take-all routing; sharp noise suppression |
| **Saturable Absorber (SA) Pure Passive** | **2.63** | 18.0% | No (100% passive) | Smooth non-zero derivative; completely resolves the gradient stall in language modeling |
| **Semiconductor Optical Amplifier (SOA)** | **2.64** | **68.0%** | **No! (Physical carrier competition)** | **Major breakthrough: Reaches 68% math accuracy purely through physical stimulated emission without an ALU** |

Detailed reports:
* **[EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)**: Deep dive into the trade-offs of Electronic vs Pure Optical vs Hybrid AGC architectures.
* **[REPORT_SA_SOA_VS_ELECTRONIC.md](REPORT_SA_SOA_VS_ELECTRONIC.md)**: In-depth analysis of Saturable Absorbers (SA) and Semiconductor Optical Amplifiers (SOA).

---

## 🛑 Three-Tier Stopping Strategy

To avoid infinite training loops and prevent overfitting, training scripts include:
1. **Target Stop**: Stops early once validation loss reaches `target_loss` (Shakespeare) or accuracy reaches `target_acc` (Math).
2. **Patience Early Stop**: Halts training if validation metrics do not improve over `patience` evaluation intervals, automatically restoring the best historical model weights.
3. **Max Iterations**: Hard upper limit on training steps.
