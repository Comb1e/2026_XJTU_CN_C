# 基因依赖性预测：从热基因到冷基因的平滑过渡

## 核心问题

当前方法在 warm genes（有训练标签，~85%）和 cold genes（无训练标签，~15%）之间有**硬边界**：
- Warm genes: μ̂_g = observed mean x̄_g
- Cold genes: μ̂_g = teacher model prediction Φ(g)
- Low-evidence genes (1-10 cell lines): 与 warm genes 同样处理，不稳定

目标：**消除硬边界，实现从热到冷的连续平滑过渡**。

---

## 文献支撑

### 1. Chronos (Dempster et al., Genome Biology 2021)

**核心公式**: r_cg = R*_cg / R_c − 1（基因敲除后生长率的分数变化）

**分层先验设计**：
- Chronos 使用 "a modified hierarchical penalty with a Gaussian kernel" 替代 CERES 的分层先验
- 基因效应向全局均值收缩，但核先验减少了选择性依赖性估计的偏差
- 冷基因（无观测）：完全依赖先验（全局均值 + 基因特征相似性）
- 低证据基因（少量观测）：先验和数据平滑混合

**关键引用**: "A more sophisticated regularization strategy that reduces bias in estimating selective dependencies while retaining effective information sharing and normalization across screens."

**来源**: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02540-7

### 2. CERES (Meyers et al., 2017)

**分层先验 + 部分池化**：
- "Combines guide efficacy estimates and copy-number correction with information sharing across cell lines via a hierarchical prior on the gene fitness effects."
- 每个基因在每个细胞系中的适应度效应向跨细胞系均值收缩
- 极限正则化下，所有基因的适应度效应移动到全局均值
- **部分池化的数学形式**: μ̂_g(c) = α · x_g(c) + (1-α) · μ_global，α = n_g/(n_g+λ)

**关键图示**: Figure 5a — 每个点是一个基因，y轴=特定细胞系中的适应度效应，x轴=跨细胞系均值，箭头=正则化惩罚的方向和幅度。

### 3. DEMETER2 (McFarland et al.)

**分层贝叶斯推断**：
- 整合细胞系筛选质量参数和分层贝叶斯推断
- 处理批次效应和可变筛选质量
- "Substantially improves estimates of gene dependency"

### 4. ashr — Adaptive Shrinkage (Stephens, 2017)

**DepMap 官方使用的效应量调节方法**：
- 在 DepMap 的两类比较（如癌症亚型 vs 其他）中，差异效应量使用 ashr 调节
- **自适应两层含义**:
  1. 先验分布 g 从数据中学习（如果 g 在零附近很尖，收缩更强）
  2. 收缩量取决于每个观测的标准误差（信息越多，收缩越少）

**R 包**: https://cran.r-project.org/package=ashr
**关键输出**: `get_pm`（后验均值，即收缩后的效应估计），`get_lfsr`（局部错误符号率）

### 5. James-Stein 估计量

**经典结果** (Stein, 1956; James & Stein, 1961):
```
δ^JS(x) = (1 − (p−2)/‖x‖²) · x
```
- 对于 p ≥ 3，James-Stein 估计量在二次损失下一致优于最大似然估计
- 这是经验贝叶斯收缩的统计基础

**Fay-Herriot 扩展**（带辅助信息）:
```
θ̂_FH = Xβ̂ + (1 − (n−p−2)/‖Z−Xβ̂‖²)(Z−Xβ̂)
```
- 先验均值由回归模型 Xβ̂ 预测（与我们的 GeneFormula 对应）
- 收缩因子自适应于数据和先验的偏差

---

## 统一平滑过渡公式

### 基因效应项 μ̂_g

```
μ̂_g = w_g · x̄_g + (1 − w_g) · Φ(g)
```

| 符号 | 含义 | 计算方式 |
|------|------|----------|
| w_g | 证据权重 | n_g / (n_g + λ) |
| n_g | 训练细胞系数 | 直接计数 (0 ~ 1140) |
| λ | 先验强度 | 经验贝叶斯估计（交叉验证或边际似然） |
| x̄_g | 观测均值 | 训练标签的 per-gene mean |
| Φ(g) | 基因先验预测 | Ridge(gene_features → y)，从 G1+G2 特征预测 |

**权重函数的连续性**：
- n_g = 1140: w_g = 1140/(1140+λ) ≈ 0.99（数据主导）
- n_g = 100: w_g = 100/(100+λ) ≈ 0.91（数据主导但略有收缩）
- n_g = 10: w_g = 10/(10+λ) ≈ 0.50（数据与先验平衡）
- n_g = 1: w_g = 1/(1+λ) ≈ 0.09（先验主导）
- n_g = 0: w_g = 0（纯先验，完全由 Φ(g) 决定）

### 细胞效应项 β̂_c

```
β̂_c = v_c · r̄_c + (1 − v_c) · Ψ(c)
```

| 符号 | 含义 | 计算方式 |
|------|------|----------|
| v_c | 证据权重 | m_c / (m_c + λ_cell) |
| m_c | 训练基因数 | 直接计数 |
| r̄_c | 观测残差均值 | 训练标签减去 μ̂_g 后的 per-cell mean |
| Ψ(c) | 细胞先验预测 | Ridge(cell_features → residual)，从 G3 特征预测 |

### 交互项 I(c,g)

```
I(c,g) = s(g) · I_formula(c,g) + (1 − s(g)) · I_transfer(c,g)
```

其中 s(g) = max_{g' ∈ warm} pathway_similarity(g, g')，即到最近热基因的通路相似度。

---

## 改进后的架构

```
训练流程：
1. 学习 Φ(g) = Ridge(G1+G2 → gene_mean_labels)
2. 学习 λ 通过最大化边际似然（或 CV）
3. 对每个基因计算 w_g = n_g/(n_g+λ)
4. μ̂_g = w_g · x̄_g + (1-w_g) · Φ(g)   ← 平滑过渡
5. 残差 r₁ = y - μ̂_g
6. 同样方式拟合 β̂_c（平滑过渡）
7. 残差 r₂ = r₁ - β̂_c
8. 拟合交互公式（module×indicator, expr_effect, etc.）
9. 对冷基因使用相似度加权的交互迁移

预测公式（对所有基因统一）：
ŷ(c,g) = [w_g·x̄_g + (1-w_g)·Φ(g)] + [v_c·r̄_c + (1-v_c)·Ψ(c)] + Blend[interactions]
```

---

## 优势

1. **无硬边界**: 冷热基因之间平滑过渡，每个基因按其证据量获得合适的处理
2. **理论基础扎实**: 经验贝叶斯/James-Stein 收缩在统计学中有 60+ 年历史，在基因组学中有 20+ 年历史
3. **DepMap/Chronos 验证**: 分层先验是 DepMap 生产流程的核心设计
4. **100% 可解释**: 每个参数（w_g, λ, Φ(g)）都有明确的统计含义
5. **自动适应**: λ 从数据中学习，不需要手动调参
6. **创新性**: 平滑过渡的方案在竞赛中具有新颖性

---

## 实现要点

### λ 的估计

选项 A — 经验贝叶斯（边际最大似然）：
```
λ̂ = argmax_λ Π_g p(x̄_g | Φ(g), n_g, λ)
```
其中 p(x̄_g | ...) 是分层模型下的边际似然。

选项 B — 交叉验证：
```python
# Grid search λ 使得 OOF 预测误差最小
for λ in [1, 5, 10, 50, 100, 500, 1000]:
    oof_scores = cross_validate_with_shrinkage(λ)
    pick λ with best score
```

选项 C — 简单启发式（快速实现）：
```
λ = median(n_g for warm genes) ≈ 1140
```
这等价于基因均值可靠性的"典型"样本量。

### 基因先验 Φ(g) 的特征选择

使用 GeneEssentialityFormula（已实现），从 G1+G2 特征预测：
- 模块成员资格（14-dim one-hot）
- 亚线粒体定位（5-dim）
- 证据权重、表达统计量（8-dim）
- 通路计数、同源标志

### 冷基因交互迁移

使用 pathway-similarity KNN（已实现 ColdGeneTransfer）：
- 基于 MitoCarta3.0 149 通路注释的余弦相似度
- K=20 最近热基因
- 权重 = 归一化相似度

---

## 参考文献

1. Dempster JM, et al. "Chronos: a cell population dynamics model of CRISPR experiments that improves inference of gene fitness effects." *Genome Biology*, 2021. https://doi.org/10.1186/s13059-021-02540-7
2. Meyers RM, et al. "Computational correction of copy number effect improves specificity of CRISPR-Cas9 essentiality screens in cancer cells." *Nature Genetics*, 2017. https://doi.org/10.1038/ng.3984
3. McFarland JM, et al. "Improved estimation of cancer dependencies from large-scale RNAi screens using model-based normalization and data integration." *Nature Communications*, 2018. https://doi.org/10.1038/s41467-018-06916-5
4. Stephens M. "False discovery rates: a new deal." *Biostatistics*, 2017. https://doi.org/10.1093/biostatistics/kxw041
5. James W, Stein C. "Estimation with quadratic loss." *Proceedings of the Fourth Berkeley Symposium*, 1961.
6. Vinceti A, et al. "A benchmark of computational methods for correcting biases of established and unknown origin in CRISPR-Cas9 screening data." *Genome Biology*, 2024. https://doi.org/10.1186/s13059-024-03336-1
7. Rath S, et al. "MitoCarta3.0: an updated mitochondrial proteome now with sub-organelle localization and pathway annotations." *Nucleic Acids Research*, 2021. https://doi.org/10.1093/nar/gkaa1011
