# RoSwish / Rotating Swish

> English is expanded by default; click **中文** below to switch to the Chinese version.

<details open>
<summary><b>English</b></summary>

Official PyTorch implementation of the paper:

> **Shengbo Zhang, Guoqiang Ren.** *RoSwish: A novel Rotating Swish activation function with adaptive rotation around zero.* **Neural Networks**, vol. 192, art. 107892, 2025.
> DOI: [10.1016/j.neunet.2025.107892](https://doi.org/10.1016/j.neunet.2025.107892) ·
> [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608025007737)
>
> **Free PDF download:** this paper is *not* open access, so the full text on ScienceDirect is behind a paywall (individual readers must purchase the article, or access it via an institutional subscription). The published PDF is archived in this repository and can be downloaded for free from the [`paper/`](paper/) directory — directly via
> [`RoSwish：A novel Rotating Swish activation function with adaptive rotation around zero.pdf`](paper/RoSwish%EF%BC%9AA%20novel%20Rotating%20Swish%20activation%20function%20with%20adaptive%20rotation%20around%20zero.pdf).

RoSwish integrates the smoothness, gradient stability, self-gating, adaptivity and noise-robustness of ReLU, GELU, ELU, PReLU and Swish into a single gate whose rotation around zero is learned channel-wise. Inspired by non-stationary stochastic process theory, the paper also proposes **NsSinAFs**, a family of non-stationary sine-wave activation functions.

## 1. RoSwish mathematical definition

RoSwish is defined as

```
f(x) = (x + α) · sigmoid(β · x) − 0.5 · α
```

where `α` and `β` are channel-wise (hidden-dimension-wise) trainable parameters that jointly regulate the **positive slope**, the **negative saturation value**, and the **negative concavity**. The function is continuously differentiable for any real-valued `(α, β)`, which facilitates gradient flow through the network.

Using the identity `tanh(x) = 2 · sigmoid(2x) − 1`, an equivalent algebraic form is

```
f(x) = x · sigmoid(β · x) + 0.5 · α · tanh(0.5 · β · x)
```

The first (simpler) form is what the code implements; the second has higher computational complexity and is only used for theoretical analysis.

Special cases:
- `α = 1, β = 1` → SiLU / Swish-1.
- For other `(α, β)` combinations RoSwish approximates a micro-rotated version of Mish near the origin, or small variants of GELU / ELU / SELU.

### Parameter initialization and optimization

| Parameter | Initial value | Shape (FC) | Shape (CNN) |
|-----------|---------------|------------|-------------|
| `α` | 0.817 | `(1, n_hidden)` | `(1, n_hidden, 1, 1)` |
| `β` | 3.000 | `(1, n_hidden)` | `(1, n_hidden, 1, 1)` |

`α` and `β` are updated jointly with the rest of the network via back-propagation (chain-rule gradients `∂f/∂α = sigmoid(βx) − 0.5`, `∂f/∂β = x·(x+α)·sigmoid(βx)·(1−sigmoid(βx))`). No range constraint is imposed during training. The paper trains all networks with the **Adam** optimizer.

> **Tip:** `α` and `β` are ordinary `nn.Parameter`s, so you can adapt them to your own task in either of two ways:
>
> - **Train them (default):** the initial values `α=0.817`, `β=3` are chosen for the relatively simple tasks in this study; feel free to pick better initial values for your setting.
> - **Freeze them (fixed RoSwish):** set `requires_grad=False` and treat RoSwish as a fixed activation. In our recent experiments we found that training `α` or `β` does **not** always yield a performance gain — a fixed pair sometimes works just as well or better.
>
> For both settings, a good starting grid is `α ∈ {0, 1, 2, 3, 4}` × `β ∈ {1, 2, 3, 4, 5}`; feel free to widen the search beyond this range.

## 2. NsSinAFs (non-stationary sine activation functions)

Inspired by non-stationary stochastic process theory, NsSinAFs replace the input `x` of any base activation `AFs(x)` by

```
f = a · x + b · sin(x)
NsSinAFs(f) = AFs(a · x + b · sin(x))
```

where `a` and `b` are channel-wise trainable parameters initialized at `1` and `0`, respectively. Intuitively, NsSinAFs inject a learnable non-stationary sine-wave perturbation into the input of an existing activation, helping the network capture time-varying / non-stationary signals. The family is evaluated on every base activation used in the paper (`NsSinReLU`, `NsSinGELU`, `NsSinSiLU`, …, `NsSinRoSwish`).

## 3. Theoretical contributions (Section 5 of the paper)

- **Universal approximation (Theorem 4 + Lemma 1):** Because `sigmoid(βx)` has an infinite Taylor expansion and is not a polynomial when `β ≠ 0`, RoSwish is non-polynomial and therefore satisfies the density condition of Leshno et al. (1993). A single-layer RoSwish network can approximate any continuous univariate function on a compact domain arbitrarily well; via the Kolmogorov–Arnold representation theorem this extends to multivariate functions and to multilayer networks. Lemma 1 guarantees the result even if some neurons degenerate to `β=0`, as long as at least one neuron keeps `β ≠ 0`.
- **Information gain (Section 5.2):** Treating the trainable parameters `θ = (α, β)` as a Gaussian random vector `N(μ, Σ)`, the information gain of RoSwish over a fixed activation is bounded by `ΔI ≥ −½ log |Σ| − C·ε > 0`. RoSwish attains the highest mutual information on ETTm2 (**0.5657 nat**) among all activations compared.
- **`β` does not collapse to 0 (Fig. 4 / Section 6.2.7):** Extending Twitter POS training to 3000 epochs, the minimum `β` converges to a strictly positive value (≈0.1 at `lr=1e-3`, ≈0.2 at `lr=1e-4`, stuck at 3 for `lr=1e-5`), empirically confirming Lemma 1.

## 4. Experimental tasks and code layout

All scripts select the activation function via the first command-line argument, e.g. `python mnist_fcn.py roswish` or `python mnist_fcn.py relu`. Supported names: `relu, gelu, silu, selu, mish, prelu, erfact, swish, psgu, aoaf, sinlu, tanhlu, aqulu, roswish`.

| Script | Task | Model / protocol | Metric |
|--------|------|------------------|--------|
| `mnist_fcn.py` | MNIST classification | 8 hidden FC layers × 128 units, 50 epochs, batch 128, Kaiming init, LRs ∈ {1e-3, 1e-4, 1e-5}, optional dropout `p` (3rd arg) | Log Loss / Error |
| `mnist_ae.py` | MNIST autoencoding | Autoencoder 1000–500–250–30–250–500–1000, 70 epochs, batch 64 | MSE |
| `deep_network_18_26_32.py` | Deep-network stability | FC networks with 18 / 26 / 32 hidden layers, MNIST, lr = 1e-4 | Log Loss / Error |
| `gaussian_cifar10.py` | CIFAR-10 noise robustness | 9-layer shallow CNN (96 channels), ZCA whitening + Gaussian noise σ=0.15, 200 epochs, batch 128, cosine-style lr schedule | Log Loss / Error |
| `load_cifar10.py` | CIFAR-10 data loader | Shared loading / ZCA utilities used by the CNN script | — |
| `twitter_pos.py` | Tweet part-of-speech tagging | 2 layers × 256 units, 30 epochs, batch 16, dropout 0.8, window size 1 over 50-d embeddings, 24 POS tags | Log Loss / Error |
| `ettm2_pred.py` | ETTm2 time-series forecasting | 8 hidden FC layers × 64 units, 50 epochs, batch 32, input length = forecast length = 96 | MSE / MAE |
| `exchange_pred.py` | Exchange-rate forecasting (Appendix A) | Daily rates of 8 countries, 1990–2016 | MSE / MAE |
| `ns_sin.py` | NsSin family on MNIST | Same 8-layer FC setup as `mnist_fcn.py`, lr = 1e-4 | Log Loss / Error |

Data:
- `data/ETTm2/ETTm2.csv` — ETTm2 dataset.
- `data/Exchange/exchange_rate.csv` — exchange-rate dataset.
- `data/Tweets/` — Twitter POS corpus (`tweets-train/dev/devtest.txt`) and 50-d embeddings (`embeddings-twitter.txt`).
- MNIST is downloaded automatically by `torchvision`; CIFAR-10 must be unpacked into `./cifar-10-batches-py/` next to the scripts.

`paper/` contains the published PDF (free to download here — the paper is not open access); `images/` contains the figures used in the paper (per-activation loss curves, α/β distribution histograms, β optimization curve, efficiency bar chart, etc.).

### Reproducibility notes

- All performance numbers in the paper are the **median of 5 runs**; best result is bold, second-best is underlined.
- The main experiments were run on an NVIDIA RTX 4090 24 GB; the accuracy-only ablation (Section 6.2.1) was run on an AMD Ryzen 5 5600H CPU.
- MNIST scripts append multiple runs (different lr / dropout) into a single `data/mnist_fcn_<af>.p` pickle keyed by `lr1/lr2/…` and `dropout1/dropout2/…`; CIFAR-10 writes `results.csv` per `(af, batch_size, lr)` directory plus a periodic `network.pth`.

## 5. Headline results (median of 5 runs)

- **MNIST classification:** RoSwish reduces Log Loss and Error by at least **16.02 %** versus the best baseline (`0.1172 → 0.0957` Log Loss, `0.0225 → 0.0185` Error at `lr=1e-3`; improvements of 8.54 %, 5.66 %, 1.82 % also hold at `lr=1e-4` and `lr=1e-5`).
- **MNIST autoencoder:** MSE reduced by at least **5 %** under low learning rate (`0.0100 → 0.0095` at `lr=1e-5`).
- **Twitter POS tagging:** improvement of at least **3.65 %** under medium-to-high LR (`0.488 → 0.4758` Log Loss at `lr=1e-3`); combining RoSwish with BatchNorm gives a further **30.19 %** reduction at low LR.
- **ETTm2 prediction:** improvement of at least **6.06 %** under medium-to-low LR (`0.1233 → 0.1200` MSE at `lr=1e-4`).
- **CIFAR-10 (Gaussian noise):** RoSwish `Log Loss = 0.4307`, `Error = 0.0841`, on par with / better than GELU (`0.4423 / 0.0837`) and SinLU (`0.4350 / 0.0833`).
- **Deep networks:** only SELU, PSGU, SinLU, tanhLU and RoSwish still converge at 26 hidden layers; at 32 layers RoSwish (`0.1481 / 0.0340`) is the best non-SELU convergence result.
- **NsSin family:** on average **+6.55 %** enhancement on MNIST (1.08 % Log-Loss reduction, 5.47 % Error reduction versus the original activations), with only a modest overhead in training time.

## 6. Installation and quick start

Requirements: Python 3+, PyTorch (with torchvision), NumPy, pandas, scikit-learn.

```bash
pip install torch torchvision numpy pandas scikit-learn
```

Run a single activation on MNIST classification:

```bash
python mnist_fcn.py roswish 1e-3 0.0
#                  af   lr    dropout
```

Run the NsSin variant on top of a base activation (see `ns_sin.py` for the exact command line):

```bash
python ns_sin.py roswish
```

## 7. Citation

If you find this work useful, please cite:

```bibtex
@article{ZHANG2025107892,
  title    = {RoSwish: A novel Rotating Swish activation function with adaptive rotation around zero},
  author   = {Shengbo Zhang and Guoqiang Ren},
  journal  = {Neural Networks},
  volume   = {192},
  pages    = {107892},
  year     = {2025},
  issn     = {0893-6080},
  doi      = {https://doi.org/10.1016/j.neunet.2025.107892},
  url      = {https://www.sciencedirect.com/science/article/pii/S0893608025007737}
}
```

The sibling function of RoSwish (Rotating SiLU) — [Bounded Sigmoid Linear Unit (B-SiLU)](https://arxiv.org/abs/2505.22074v1):

```bibtex
@article{bsilu,
  title   = {The Resurrection of the ReLU},
  author  = {Horuz, Co{\c{s}}ku Can and Kasenbacher, Geoffrey and Higuchi, Saya and Kairat, Sebastian and Stoltz, Jendrik and Pesl, Moritz and Moser, Bernhard A and Linse, Christoph and Martinetz, Thomas and Otte, Sebastian},
  journal = {arXiv preprint arXiv:2505.22074},
  year    = {2025}
}
```

## 8. Contact

Questions or usage of the code: `zhangshengbo@sdust.edu.cn`.

</details>

<details>
<summary><b>中文</b></summary>

本仓库为论文的官方 PyTorch 实现：

> **Shengbo Zhang, Guoqiang Ren.** *RoSwish: A novel Rotating Swish activation function with adaptive rotation around zero.* **Neural Networks**, vol. 192, art. 107892, 2025.
> DOI: [10.1016/j.neunet.2025.107892](https://doi.org/10.1016/j.neunet.2025.107892) ·
> [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608025007737)
>
> **免费下载 PDF：** 本论文为**非开放获取（非 OA）**状态，ScienceDirect 上的全文位于付费墙后（个人读者需按篇购买，或通过机构订阅访问）。论文正式发表版 PDF 已随本仓库存档于 [`paper/`](paper/) 目录，可直接免费下载：
> [`RoSwish：A novel Rotating Swish activation function with adaptive rotation around zero.pdf`](paper/RoSwish%EF%BC%9AA%20novel%20Rotating%20Swish%20activation%20function%20with%20adaptive%20rotation%20around%20zero.pdf)。

RoSwish 将 ReLU、GELU、ELU、PReLU、Swish 等激活函数的平滑性、梯度稳定性、自门控、自适应性与噪声鲁棒性整合进同一个门控机制中，其围绕零点的旋转角度按通道维度自适应学习。受非平稳随机过程理论启发，论文还提出了 **NsSinAFs**——一族基于非平稳正弦波的激活函数。

## 1. RoSwish 数学定义

RoSwish 定义为：

```
f(x) = (x + α) · sigmoid(β · x) − 0.5 · α
```

其中 `α` 和 `β` 是按通道（隐藏层维度）逐元素训练的参数，共同调节**正斜率**、**负饱和值**与**负凹性**。对任意实数 `(α, β)`，RoSwish 处处连续可微，有利于梯度在网络中传播。

利用恒等式 `tanh(x) = 2 · sigmoid(2x) − 1`，其等价代数形式为：

```
f(x) = x · sigmoid(β · x) + 0.5 · α · tanh(0.5 · β · x)
```

代码中实现的是第一个（更简洁的）形式；第二个形式计算复杂度更高，仅用于理论分析。

特殊情形：
- `α = 1, β = 1` → SiLU / Swish-1。
- 取其他 `(α, β)` 时，RoSwish 可近似为原点附近微旋转的 Mish，或与 GELU / ELU / SELU 相差很小的变体。

### 参数初始化与优化

| 参数 | 初值 | FC 中的形状 | CNN 中的形状 |
|------|------|-------------|--------------|
| `α` | 0.817 | `(1, n_hidden)` | `(1, n_hidden, 1, 1)` |
| `β` | 3.000 | `(1, n_hidden)` | `(1, n_hidden, 1, 1)` |

`α`、`β` 通过反向传播与网络其他参数一同用链式法则更新（梯度 `∂f/∂α = sigmoid(βx) − 0.5`，`∂f/∂β = x·(x+α)·sigmoid(βx)·(1−sigmoid(βx))`），训练过程中不对取值范围施加约束。论文中所有网络均使用 **Adam** 优化器训练。

> **提示：** `α`、`β` 是普通的 `nn.Parameter`，你可以在自己的任务上按以下两种方式之一调整：
>
> - **训练它们（默认）：** 初值 `α=0.817`、`β=3` 是针对本文较简单任务选定的，可根据自己的任务选择更好的初值。
> - **冻结它们（固定 RoSwish）：** 设置 `requires_grad=False`，把 RoSwish 当作固定激活函数来用。在近期实验任务中我们发现，训练 `α` 或 `β` **并不总是带来性能增益**，此时固定参数反而更优。
>
> 两种方式都建议先用网格 `α ∈ {0, 1, 2, 3, 4}` × `β ∈ {1, 2, 3, 4, 5}` 起步，范围之外也可以自行扩大搜索。

## 2. NsSinAFs（非平稳正弦激活函数族）

受非平稳随机过程理论启发，NsSinAFs 将任意基础激活 `AFs(x)` 的输入 `x` 替换为：

```
f = a · x + b · sin(x)
NsSinAFs(f) = AFs(a · x + b · sin(x))
```

其中 `a`、`b` 是按通道逐元素训练的参数，初值分别为 `1` 和 `0`。直观上，NsSinAFs 向已有激活的输入端注入了一个可学习的非平稳正弦扰动，帮助网络捕捉时变 / 非平稳信号。论文对所有基础激活都构造了对应变体（`NsSinReLU`、`NsSinGELU`、`NsSinSiLU`、……、`NsSinRoSwish`）。

## 3. 理论贡献（论文第 5 节）

- **通用逼近能力（Theorem 4 + Lemma 1）：** 由于 `sigmoid(βx)` 的 Taylor 展开含无穷多项、在 `β ≠ 0` 时不是多项式，RoSwish 非多项式，满足 Leshno 等人（1993）的稠密性条件。单隐层 RoSwish 网络可以在紧集上以任意精度逼近任意连续一元函数；结合 Kolmogorov–Arnold 表示定理可推广到多元情形与多层网络。Lemma 1 进一步保证：即使部分神经元退化到 `β=0`，只要至少一个神经元保持 `β ≠ 0`，逼近能力仍然成立。
- **信息增益（第 5.2 节）：** 将可训练参数 `θ = (α, β)` 视为高斯随机向量 `N(μ, Σ)`，RoSwish 相对固定激活的信息增益有下界 `ΔI ≥ −½ log|Σ| − C·ε > 0`。在 ETTm2 上 RoSwish 取得所有对比激活中最高的互信息（**0.5657 nat**）。
- **`β` 不会坍塌到 0（Fig. 4 / 第 6.2.7 节）：** 把 Twitter POS 训练延长到 3000 个 epoch，`β` 的最小值收敛到严格为正的数（`lr=1e-3` 时约 0.1，`lr=1e-4` 时约 0.2，`lr=1e-5` 时停在 3），从实验上验证了 Lemma 1 的猜想。

## 4. 实验任务与代码结构

所有脚本通过命令行第一个参数选择激活函数，例如 `python mnist_fcn.py roswish` 或 `python mnist_fcn.py relu`。可选名称：`relu, gelu, silu, selu, mish, prelu, erfact, swish, psgu, aoaf, sinlu, tanhlu, aqulu, roswish`。

| 脚本 | 任务 | 模型 / 实验设置 | 指标 |
|------|------|------------------|------|
| `mnist_fcn.py` | MNIST 分类 | 8 层全连接 × 128 单元，50 epoch，batch 128，Kaiming 初始化，学习率 ∈ {1e-3, 1e-4, 1e-5}，可选 dropout（第 3 个参数） | Log Loss / Error |
| `mnist_ae.py` | MNIST 自编码 | 自编码器 1000–500–250–30–250–500–1000，70 epoch，batch 64 | MSE |
| `deep_network_18_26_32.py` | 深层网络稳定性 | 18 / 26 / 32 层全连接，MNIST，lr = 1e-4 | Log Loss / Error |
| `gaussian_cifar10.py` | CIFAR-10 噪声鲁棒性 | 9 层浅层 CNN（96 通道），ZCA 白化 + 高斯噪声 σ=0.15，200 epoch，batch 128，余弦式学习率调度 | Log Loss / Error |
| `load_cifar10.py` | CIFAR-10 数据加载 | CNN 脚本共用的加载 / ZCA 工具 | — |
| `twitter_pos.py` | Twitter 词性标注 | 2 层 × 256 单元，30 epoch，batch 16，dropout 0.8，窗口大小 1，使用 50 维词向量，24 个词性标签 | Log Loss / Error |
| `ettm2_pred.py` | ETTm2 时间序列预测 | 8 层全连接 × 64 单元，50 epoch，batch 32，输入长度 = 预测长度 = 96 | MSE / MAE |
| `exchange_pred.py` | 汇率预测（附录 A） | 8 个国家 1990–2016 年日汇率 | MSE / MAE |
| `ns_sin.py` | MNIST 上的 NsSin 族 | 与 `mnist_fcn.py` 相同的 8 层全连接设置，lr = 1e-4 | Log Loss / Error |

数据：
- `data/ETTm2/ETTm2.csv` — ETTm2 数据集。
- `data/Exchange/exchange_rate.csv` — 汇率数据集。
- `data/Tweets/` — Twitter 词性标注语料（`tweets-train/dev/devtest.txt`）与 50 维词向量（`embeddings-twitter.txt`）。
- MNIST 由 `torchvision` 自动下载；CIFAR-10 需要手动解压到脚本同级目录的 `./cifar-10-batches-py/`。

`paper/` 为已发表论文 PDF（本仓库可免费下载——论文为非 OA 状态）；`images/` 为论文所用图（各激活函数 loss 曲线、α/β 分布直方图、β 优化曲线、训练耗时与内存柱状图等）。

### 复现说明

- 论文中所有性能数字均为 **5 次运行的中位数**；最优结果加粗，次优结果加下划线。
- 主实验在 NVIDIA RTX 4090 24 GB 上完成；仅精度对比的消融实验（第 6.2.1 节）在 AMD Ryzen 5 5600H CPU 上完成。
- MNIST 脚本会把多次实验（不同 lr / dropout）按 `lr1/lr2/…`、`dropout1/dropout2/…` 为键追加写进同一个 `data/mnist_fcn_<af>.p` pickle；CIFAR-10 则按 `(af, batch_size, lr)` 目录写 `results.csv`，并定期保存 `network.pth`。

## 5. 主要实验结果（5 次运行中位数）

- **MNIST 分类：** 相对最优基线，Log Loss 与 Error 平均下降至少 **16.02 %**（`lr=1e-3` 下 `0.1172 → 0.0957`，`0.0225 → 0.0185`；在 `lr=1e-4`、`lr=1e-5` 下仍分别有 8.54 %、5.66 %、1.82 % 的下降）。
- **MNIST 自编码：** 低学习率下 MSE 至少降低 **5 %**（`lr=1e-5` 下 `0.0100 → 0.0095`）。
- **Twitter 词性标注：** 中高学习率下提升至少 **3.65 %**（`lr=1e-3` 下 Log Loss `0.488 → 0.4758`）；RoSwish 与 BatchNorm 结合后，低学习率下再降 **30.19 %**。
- **ETTm2 预测：** 中低学习率下提升至少 **6.06 %**（`lr=1e-4` 下 MSE `0.1233 → 0.1200`）。
- **CIFAR-10（高斯噪声）：** RoSwish `Log Loss = 0.4307`，`Error = 0.0841`，与 GELU（`0.4423 / 0.0837`）、SinLU（`0.4350 / 0.0833`）持平或更优。
- **深层网络：** 26 层时仅 SELU、PSGU、SinLU、tanhLU、RoSwish 仍收敛；32 层时 RoSwish（`0.1481 / 0.0340`）是除 SELU 外最好的收敛结果。
- **NsSin 族：** 在 MNIST 上相对原激活平均提升 **6.55 %**（Log Loss 降 1.08 %，Error 降 5.47 %），训练时间仅有少量增加。

## 6. 安装与快速开始

依赖：Python 3+、PyTorch（含 torchvision）、NumPy、pandas、scikit-learn。

```bash
pip install torch torchvision numpy pandas scikit-learn
```

在 MNIST 分类上跑某个激活：

```bash
python mnist_fcn.py roswish 1e-3 0.0
#                  激活名  学习率  dropout
```

在基础激活之上跑 NsSin 变体（具体命令行参数见 `ns_sin.py`）：

```bash
python ns_sin.py roswish
```

## 7. 引用

如果本工作对你的研究有帮助，请引用：

```bibtex
@article{ZHANG2025107892,
  title    = {RoSwish: A novel Rotating Swish activation function with adaptive rotation around zero},
  author   = {Shengbo Zhang and Guoqiang Ren},
  journal  = {Neural Networks},
  volume   = {192},
  pages    = {107892},
  year     = {2025},
  issn     = {0893-6080},
  doi      = {https://doi.org/10.1016/j.neunet.2025.107892},
  url      = {https://www.sciencedirect.com/science/article/pii/S0893608025007737}
}
```

RoSwish 的姊妹函数（Rotating SiLU）——[Bounded Sigmoid Linear Unit (B-SiLU)](https://arxiv.org/abs/2505.22074v1)：

```bibtex
@article{bsilu,
  title   = {The Resurrection of the ReLU},
  author  = {Horuz, Co{\c{s}}ku Can and Kasenbacher, Geoffrey and Higuchi, Saya and Kairat, Sebastian and Stoltz, Jendrik and Pesl, Moritz and Moser, Bernhard A and Linse, Christoph and Martinetz, Thomas and Otte, Sebastian},
  journal = {arXiv preprint arXiv:2505.22074},
  year    = {2025}
}
```

## 8. 联系方式

如有问题或希望使用本代码，请联系：`zhangshengbo@sdust.edu.cn`。

</details>
