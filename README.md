# OptiCNN: 光学神经网络与光子计算物理仿真

> 用 Python 探索光子计算（Optical Computing）的物理可行性与现实瓶颈。

---

## 📌 项目概述

本项目旨在脱离复杂的黑盒深度学习框架，从**麦克斯韦电磁波方程与干涉光学原理**出发，用纯代码直观感受：
1. **光如何天然做线性代数**：通过马赫-曾德尔干涉仪 (MZI) 的波导耦合与相位调节，光速完成乘法与正余弦分配。
2. **光电转换断点与物理瓶颈 (O/E-DSP-E/O)**：为什么“光计算在矩阵乘法快如闪电，但在非线性激活层却受制于电芯片”？

---

## 📁 目录结构

```text
d:\opticnn
├── step2_mzi_hello_world.py        # Step 2: 跑通光的“Hello World” (2x2 MZI 干涉与乘法)
├── step2_mzi_plot.py               # Step 2 可视化: 生成干涉能量分配曲线图
├── step3_optoelectronic_bottleneck.py # Step 3: 光电转换断点 (O/E -> Softmax -> E/O 损耗审计)
├── step2_neuroptica_demo.py        # (可选) 使用 Stanford neuroptica 库的 Clements 拓扑演示
├── .gitignore                      # Git 忽略配置
└── README.md                       # 本文档
```

---

## 🚀 快速上手

### 1. 环境准备
```bash
pip install numpy neuroptica matplotlib
```

### 2. 运行 Step 2: 光的 "Hello World" (2x2 MZI)
```bash
python step2_mzi_hello_world.py
```
**终端输出亮点**：
- 输入端口 1 注入单色连续光波 $E_{in} = [1+0j, 0+0j]^T$。
- 经过定向耦合分束器 (50:50 Directional Coupler) 与移相器 $\Delta\phi$。
- 实时终端能量条直观展示两路输出光强 $|E_1|^2$ 和 $|E_2|^2$ 完美跟随 $\sin^2(\Delta\phi/2)$ 与 $\cos^2(\Delta\phi/2)$ 进行光速级的线性乘法与分配！
- 始终保持总光强守恒 $I_1 + I_2 = 1.0$。

还可以生成高分辨率干涉曲线图：
```bash
python step2_mzi_plot.py
```

### 3. 运行 Step 3: 模拟“光电转换”现实瓶颈
```bash
python step3_optoelectronic_bottleneck.py
```
**关键实验过程**：
1. **Layer 1 (光域线性乘法)**：光子以 $100,000 \text{ km/s}$ 的速度飞跃 MZI 矩阵，延迟仅数十皮秒 (ps)，功耗极低 (< 0.1 fJ/MAC)。
2. **O/E 断点 (光电探测与 ADC)**：
   - 光电探测器 (PD) 是平方律探测，只输出实数光强 $I = |E|^2$。
   - **原本携带所有波前干涉信息的复数相位全部丢失！**
   - 8-bit ADC 模数转换引入量化噪声，并消耗数皮焦 (pJ) 能量，产生数纳秒 (ns) 延迟。
3. **电域 Softmax**：在传统电子 ALU / GPU 中跑标准的实数 Softmax。
4. **E/O 断点 (DAC 与电光调制器 MZM)**：
   - 必须使用全新的激光光源，强行将实数重编码为光波（原先的物理相干性已彻底破坏，相位被强行置零或伪造）。
   - 电光调制器产生 3~5 dB 的严重插入损耗（光功率直接衰减一半以上）。
5. **审计报告对比**：
   - 延迟：光域 ~30 ps vs 光电转换 ~3000 ps (慢了近 100 倍！)
   - 能耗：光域 < 0.1 fJ vs 光电转换 ~20 pJ (高了 1000 倍！)
   - 光功率损失：每个光电循环带来高达 6~10 dB 的光能衰减。

---

## 🔗 关于 Git 仓库设置说明

你可以随时在本地管理版本，若需推送到远程（如 GitHub / GitLab / Gitee）：

1. 本地仓库已通过 `git init` 完成初始化。
2. 在 GitHub 上新建一个空仓库（例如名为 `opticnn`，**不要**勾选自动创建 README）。
3. 复制远程仓库地址，在当前项目目录下执行：
   ```bash
   git add .
   git commit -m "feat: init optical neural network simulation (step 2 & step 3)"
   git branch -M main
   git remote add origin https://github.com/<你的用户名>/opticnn.git
   git push -u origin main
   ```
