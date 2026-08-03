# 归纳矩阵补全 (Inductive Matrix Completion) 用于基因依赖性预测

## 核心问题

当前的 AMMI-style SVD 双线性交互模型：
```
ŷ(c,g) = μ̂_g + β̂_c + Σ_k σ_k · u_k(c) · v_k(g)
```

**致命缺陷**：冷基因的 v_k(g) 无法从 G1+G2 特征预测（特征太弱），导致冷基因分数崩溃到 S≈44（之前是 S≈67）。

## 突破性方案：归纳矩阵补全 (IMC)

### 数学形式

```
ŷ(c,g) = x_c^T Z y_g
```

其中：
- **x_c** ∈ ℝ^{f_c}：细胞 c 的特征向量（G3: ~115 维，包括 14 indicator + 29 lineage + 20 pathway PCA + 10 derived + 42 raw scores）
- **y_g** ∈ ℝ^{f_g}：基因 g 的特征向量（G1+G2: ~50 维，包括 14 module + 5 location + 6 evidence + 8 expression profile + pathway count）
- **Z** = WH^T ∈ ℝ^{f_c × f_g}：双线性交互核，低秩分解为 W ∈ ℝ^{f_c × r}, H ∈ ℝ^{f_g × r}

### 参数规模

设 f_c = 115, f_g = 50, r = 10:
- Z 的满秩参数: 115 × 50 = 5,750
- W + H 的低秩参数: (115 + 50) × 10 = 1,650
- 当前公式模型总参数: ~120 (gene prior) + ~120 (cell prior) + 12×2190 (SVD) ≈ 26,000
- IMC 参数更少，泛化更好

### 核心优势

1. **天然冷启动**：冷基因预测使用 y_g（特征对所有基因都存在），不需要任何标签
2. **统一模型**：不需要分别建模 μ̂_g、β̂_c、I(c,g) — 一个双线性形式同时捕获主效应和交互效应
3. **完全公式可解释**：Z 矩阵的每个元素 Z_ij 代表"细胞特征 i 与基因特征 j 的交互系数"
4. **低秩正则化**：Z = WH^T 自动防止过拟合
5. **理论基础扎实**：IMC 在 lncRNA-disease association、drug-target interaction、推荐系统等领域广泛验证

### 优化问题

```
min_{W,H}  Σ_{(c,g)∈obs} (y_{cg} - x_c^T W H^T y_g)^2 + λ_w ||W||_F^2 + λ_h ||H||_F^2
```

可通过交替最小二乘 (ALS) 求解：
1. 固定 H，解 W：Ridge 回归
2. 固定 W，解 H：Ridge 回归
3. 交替直到收敛

### 与当前模型的对比

| 属性 | 当前 AMMI-SVD | IMC |
|------|-------------|-----|
| 冷基因处理 | v_k 需从特征预测（弱） | 直接用基因特征 y_g |
| 参数数 | ~26,000 | ~1,650 |
| 可解释性 | 需解释 SVD 成分 | 直接可解释（特征×特征交互） |
| 主效应 | 单独建模 μ̂_g, β̂_c | 统一在双线性形式中 |
| 冷基因分数 | S≈44 | 预期 S≈65-70 |

### 实现复杂度

- 核心算法：交替最小二乘（~50 行 Python）
- 不需要新的依赖库
- 训练时间：~10 秒（vs 当前 ~2 分钟 CV）

## 相关文献

1. **Jain & Dhillon (2013)**: "Provable Inductive Matrix Completion." *arXiv:1306.0626*.
   - 原始 IMC 理论：min_{Z: rank(Z)≤r} ||P_Ω(Y - X_c Z X_g^T)||_F^2
   - 证明在一定条件下可恢复真实低秩矩阵

2. **Natarajan & Dhillon (2014)**: "Inductive matrix completion for predicting gene–disease associations." *Bioinformatics*.
   - 首次将 IMC 应用于基因-疾病关联预测
   - 使用基因表达 PCA 和疾病文本特征作为 side information
   - 冷启动预测显著优于标准矩阵补全

3. **MM-LDA (2023)**: "Graph Attention Networks + IMC for lncRNA-disease prediction." *Genomics*.
   - AUC=0.9395, AUPR=0.8057
   - IMC explicitly solves cold-start problem

4. **deepDGA (2022)**: "Heterogeneous network + deep Transformer + IMC for disease-gene prediction." *IEEE*.
   - Meta-path Transformer for node features, then IMC for prediction

5. **Hastie, Mazumder, et al. (2015)**: "Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares." *JMLR*.
   - softImpute R package — 核范数正则化的矩阵补全
   - ALS 算法的高效实现

6. **Chronos (Dempster et al., 2021)**: "Cell population dynamics model of CRISPR experiments." *Genome Biology*.
   - 分层高斯核先验 = 经验贝叶斯 + 特征空间的低秩结构

## 与本文献相关的项目文件

- `src/prediction/formula.py`: 当前 SVD 双线性交互实现
- `src/prediction/baselines.py`: 基因教师模型和细胞基线
- `src/prediction/features.py`: G1-G5 特征工程（细胞特征和基因特征）
- `参考文献/interaction_term_optimization_research.md`: 之前的交互项优化研究
