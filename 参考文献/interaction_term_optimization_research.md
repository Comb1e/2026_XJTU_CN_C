# 交互项优化研究：突破 R²=0.0018 瓶颈

## 当前问题

当前四个交互公式合计 R²≈0.0018（在移除基因主效应 μ̂_g 和细胞主效应 β̂_c 后的双残差上）：
- I_mod: Module_m(g) × Indicator_m(c) — 14 coeffs
- I_expr: θ₁z + θ₂|z| + θ₃max(0,z) + θ₄min(0,z) — 4 coeffs
- I_match: Module_m(g) × ExprPercentile(c,g) — 14 coeffs
- I_ew: ω × EvidenceWeight(g) × z_cg — 1 coeff

总共33个可学习参数，过于刚性，无法捕获丰富的 gene×cell 交互结构。

残差矩阵 R_{cg} 尺寸 ~1140 cells × ~1140 genes，残差方差占比 ~27%。
30个参数试图拟合 ~1.3M 个残差 — 模型容量严重不足。

---

## 文献调研

### 1. AMMI / Bilinear Models (Gollob 1968; Gauch 1988)

**核心公式**:
```
Y_{ij} = μ + G_i + E_j + Σ_k λ_k · γ_{ik} · δ_{jk} + ε_{ij}
```

即: **加性主效应 + SVD分解的乘性交互项**。

- GGE biplot: Y - E_j = G_i + GE_{ij}，对 G_i + GE_{ij} 做 SVD
- AMMI: 对加性残差 E_{ij} = Y_{ij} - μ - G_i - E_j 做 SVD
- 前几个乘性项通常捕获大部分交互平方和
- 使用 F-test 或参数 bootstrap 决定保留的 SVD 成分数

**R 包**: `Bilinear`, `agricolae`

**应用于本项目**:
```
ŷ(c,g) = μ̂_g + β̂_c + Σ_{k=1}^K σ_k · u_k(c) · v_k(g)
```
其中 σ_k, u_k(c), v_k(g) 来自残差矩阵 R_{cg} 的 SVD 分解。

**来源**: https://rdrr.io/github/nsantantonio/Bilinear/

### 2. 贝叶斯低秩矩阵分解 — phenix (Dahl et al.)

**核心公式**:
```
Y = U + ε
U = Sβ  (low-rank matrix, rank M ≪ min(N,P))
S ∼ MatrixNormal(0, K, I_M)
β ∼ MatrixNormal(0, D, B)
```

**关键性质 — Model-Induced Regularization (MIR)**:
即使使用无信息先验，U 的诱导后验也能自动正则化：
```
σᵢ(Û) = max((1 − Nσ²/σᵢ(Y)²) · σᵢ(Y), 0)
```
这等价于对奇异值的 **James-Stein 收缩**，自动选择秩。

**应用于本项目**: 用 Bayesian 低秩分解做 denoising，提取干净的交互信号。

**来源**: bioRxiv (phenix model paper)

### 3. CRISPR Screen SVD 残差校正

CRISPR 筛选数据的标准处理流程中，SVD 用于去除控制筛选残差矩阵中的优势协变模式：

- 对对照-guide 残差矩阵做 SVD
- 移除前 k 个奇异向量以净化稀疏生物信号
- k 的选择通过 FLEX precision-recall 在通路/复合体重建上的客观基准确定

**应用于本项目**: 反其道而行 — 保留前 k 个奇异向量（交互信号），去掉噪声尾。

**来源**: bioRxiv 2025 CRISPR screens pipeline

### 4. Pint — 加性 + 成对交互 Lasso

**核心公式**:
```
y_i = Σ_j X_{i,j} · effect(j) + Σ_{k>j} effect(j,k) + noise
```

通过并行 coordinate-descent lasso 估计所有成对交互效应。

**关键发现**: 在模拟数据上 R²≈0.99，在真实 InfectX fitness 数据上拟合很差 —
说明真实生物系统的 fitness 不是简单的加性+成对交互。

**教训**: 不要盲目增加交互参数 — 需要受生物先验约束的结构化交互。

**来源**: bioRxiv 2021 (Pint method paper)

### 5. 功能基因组中的 ANOVA 方差分解

在酵母孢子形成基因表达 QTN 映射中：
- 总方差解释: 15-29%（中位数）
- 主效应: 0-8.5%（中位数）
- **交互效应: 12-16%（中位数）— 超过主效应**

说明交互效应在生物系统中确实可以解释大量方差，但需要正确的模型结构。

**来源**: PLOS Genetics 2014 (sporulation QTN paper)

### 6. 线粒体基因表达的上下文特异性

- 线粒体-核共表达在不同组织间存在正/负相关性反转
- 不同 MitoPathway 在不同组织中有不同的共表达模式
- 特定的核转录因子（p53, MYC, NRF-1, NRF-2）同时调控线粒体和核基因表达
- mtDNA cis-eQTL 效应具有组织特异性

**生物学含义**: 交互效应不是随机的 — 它受组织谱系、表达上下文、以及特定转录程序的调控。

**来源**: ScienceDirect 2022 (mitonuclear coordination review); PMC 2024 (MitoPathway scores)

---

## 技术方案

### 方案 A: SVD 双线性交互 (AMMI-style) ★★★★★ 推荐

**数学形式**:
```
I(c,g) = Σ_{k=1}^K σ_k · u_k(c) · v_k(g)
```
其中 UΣV^T = SVD(R_{cg})，R_{cg} = y_{cg} - μ̂_g - β̂_c。

**可解释性**:
- 每个 k 是一个 "交互模式" — 一组基因在特定细胞上下文中共同变化的模式
- σ_k 是该模式的强度
- u_k(c) 是该模式下细胞 c 的得分（可追溯生物学意义）
- v_k(g) 是该模式下基因 g 的载荷（可做 GO 富集分析）

**公式复杂度**: K× (1140 + 1140 + 1) 个参数 — 远超 33 个，但仍可解释

**实现**:
```python
# 构建残差矩阵
R = np.zeros((n_cells, n_genes))
for (c_i, g_j), residual in zip(pairs, r2):
    R[cell_idx[c_i], gene_idx[g_j]] = residual

# SVD
U, S, Vt = np.linalg.svd(R, full_matrices=False)

# 用前 K 个成分做交互预测
I_cg = U[:, :K] @ np.diag(S[:K]) @ Vt[:K, :]
```

**K 的选择**: 通过交叉验证（按基因分组）确定最优 K。

**潜在问题**: 冷基因的 v_k(g) 未知（残差矩阵中该列全缺失）。
**解决**: v_k(g) 从基因特征预测（Ridge(v_k → G1+G2)）。

### 方案 B: 逐模块表达响应曲线 (Module-Specific Splines)

**数学形式**:
```
I_expr_mod(c,g) = Σ_m Module_m(g) · f_m(z_{c,g})
```
其中 f_m 是模块 m 的平滑函数（三次样条基，5-7 个节点）。

**对比当前 I_expr**: 当前使用 4 个全局基函数（z, |z|, max(0,z), min(0,z)），
无法捕获不同模块对表达的差异化响应。

**公式复杂度**: 14 modules × 5 spline basis = 70 coeffs

**可解释性**: 每个模块的 f_m(z) 可绘制为曲线图。

### 方案 C: 149 通路分辨率交互

**数学形式**:
```
I_pw_match(c,g) = α · PW_mean(c,g) + β · PW_max(c,g)
```
其中 PW_mean(c,g) = gene g 所注释的 149 MitoPathway 在 cell c 中的平均得分。

**对比当前 I_match**: 当前使用粗粒度的 14 模块，丢失了通路分辨率的精细信息。

**公式复杂度**: 2 coeffs

### 方案 D: 谱系条件交互

**数学形式**:
```
I_lineage_mod(c,g) = Σ_l Σ_m η_{l,m} · Lineage_l(c) · Module_m(g) · Indicator_m(c)
```
谱系 × 模块的三阶交互。

**公式复杂度**: N_lineages × 14 coeffs ≈ 29 × 14 = 406 coeffs
可通过 Ridge 正则化压缩。

---

## 推荐优化路线

### Phase 1: SVD 双线性交互（核心提升）

在残差矩阵 R_{cg} 上做 SVD，使用前 K 个成分作为交互项：

```
ŷ(c,g) = μ̂_g + β̂_c + Σ_{k=1}^K σ_k · u_k(c) · v_k(g)
```

对新细胞/基因的扩展：预计算 u_k(c) 和 v_k(g)，冷基因的 v_k(g) 通过 Ridge(gene_features → v_k) 预测。

**预期提升**: R² 从 0.0018 → 0.05-0.10（保守估计，基于 SVD 通常能捕获残差方差的 20-40%）

**K 取值**: 5-20，通过 gene-grouped CV 确定

### Phase 2: 逐模块表达响应 + SVD 残差

在 SVD 交互的基础上，对剩余的稀疏信号使用逐模块样条：

```
I_total = SVD_interaction + ModuleSplineEffect
```

### Phase 3: 149 通路特征 + 谱系交互

将通路分辨率和谱系信息纳入交互模型。

---

## 参考文献

1. Gollob HF. "A statistical model which combines features of factor analytic and analysis of variance techniques." *Psychometrika*, 1968.
2. Gauch HG. "Model selection and validation for yield trials with interaction." *Biometrics*, 1988.
3. Gheorghe M, Hart T. "Optimal construction of a functional interaction network from pooled library CRISPR fitness screens." *BMC Bioinformatics*, 2022.
4. Stephens M. "False discovery rates: a new deal." *Biostatistics*, 2017. (ashr adaptive shrinkage)
5. Dempster JM, et al. "Chronos: a cell population dynamics model of CRISPR experiments." *Genome Biology*, 2021.
6. Dahl A, et al. "A multi-phenotype linear mixed model with Bayesian low-rank matrix factorization." (phenix)
7. PLOS Genetics 2014 — Sporulation QTN effects on gene expression (ANOVA variance decomposition)
8. "Coordination of mitochondrial and nuclear gene-expression regulation in health, evolution, and disease." *Current Opinion in Genetics & Development*, 2022.
