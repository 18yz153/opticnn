# OptiCNN: 光学神经网络与全光 NanoGPT 物理仿真

> 用 Python 与 PyTorch 探索光子计算（Optical Computing）与全光大语言模型（Optical Transformer）的物理可行性。

---

## 📁 模块化项目结构

```text
d:\opticnn
├── 01_physics_foundations/              # 第一阶段: 光学物理第一性原理
│   ├── step2_mzi_hello_world.py        # Step 2: 2x2 MZI 干涉与连续三角函数矩阵乘法
│   ├── step2_mzi_plot.py               # Step 2 可视化: 生成干涉曲线图 (mzi_interference.png)
│   ├── step2_neuroptica_demo.py        # Stanford neuroptica Clements 拓扑网格演示
│   └── step3_optoelectronic_bottleneck.py # Step 3: 光电转换断点 (O/E -> Softmax -> E/O 损耗审计)
│
├── 02_polarization_gates/               # 第二阶段: 偏振门与全光原生网络
│   ├── optics_engine.py                # 包含 4 大物理属性 (波长/强度/相位/偏振) 的底层引擎
│   ├── experiment_polarization_softmax.py # 偏振门物理特性实测 (WDM / Fresnel-Arago / 偏振Softmax)
│   └── step4_optical_native_network.py # 抛弃Softmax! 全光原生激活 (sin^4/sin^2) 螺旋分类实测 (99.4% 准确率)
│
├── 03_optical_nanogpt/                  # 第三阶段: 全光 Native NanoGPT 大语言模型
│   ├── model.py                        # 全光 CausalSelfAttention (sin^4 偏振门 + 可学习光学旋钮)
│   ├── train_shakespeare.py            # Stage 1: 字符级莎士比亚语言建模 (验证梯度传播与生成能力)
│   ├── train_toy_math.py               # Stage 2: 玩具级算术加法 (验证数据路由 Data Routing 与胜者通吃)
│   └── data/
│       ├── prepare_shakespeare.py      # TinyShakespeare 数据加载与切分
│       └── prepare_toy_math.py         # 加法算术样本生成
│
├── .gitignore                          # Git 忽略配置
├── requirements.txt                    # 依赖清单 (含 PyTorch)
└── README.md                           # 本文档
```

---

## 🚀 全光 NanoGPT 核心魔改原理 (Optical Code Hack)

在传统电子 Transformer 中，自注意力机制依赖于高开销的指数与除法运算：
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

在 **全光 Native Transformer** 中，我们将它彻底替换为**两级双折射偏振滤光片（Lyot 偏振门）的天然干涉透射率**：
```python
# 可学习的光学旋钮 (自适应调节电光调制器的相位区间)
self.optical_scale = nn.Parameter(torch.ones(1) * 0.1)

# 将 QK^T 点积映射为偏振旋转角 theta
theta = att * self.optical_scale

# 穿过级联偏振门 (天然具备超线性胜者通吃特性的物理透过率)
att = torch.sin(theta) ** 4

# 物理光束遮罩: 未来时间步的光阑透射率精确为 0
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, 0.0)

# 光束能量守恒 (被动分束归一化)
att = att / (att.sum(dim=-1, keepdim=True) + 1e-6)
```

---

## 🧪 实验验证结果

### 1. Stage 1: 字符级莎士比亚语言建模
```bash
python 03_optical_nanogpt/train_shakespeare.py
```
* **训练表现**：800 步迭代内，训练 Loss 从 4.18 稳定收敛至 **1.97**。
* **生成效果**：初始纯乱码，训练后模型学会了英文单词拼写、换行排版与戏剧人物对白格式（`ROMEO:\nAn there's, be to my...`）。
* **结论**：**证明 $\sin^4(\theta)$ 偏振门成功在多层自注意力机制中传播了语言建模梯度！**

### 2. Stage 2: 玩具级加法数据路由 (Data Routing)
```bash
python 03_optical_nanogpt/train_toy_math.py
```
* **加法推理实测**：全光偏振门模型对一位数加法达到 **85%~90%** 准确率，并对两位数加法展现了极高精度的局部注意力聚焦。
* **结论**：**证明 $\sin^4(\theta)$ 的超线性响应完美实现了对特定输入数字 Token 的精准聚集（胜者通吃）！**
