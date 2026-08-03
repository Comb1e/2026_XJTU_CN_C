# 冷基因预测优化方案：目标 S ≥ 80

## 一、现状诊断

### 1.1 当前性能（200 genes validation）

| 指标 | 全量 | Warm | Cold |
|------|------|------|------|
| Final S | 66.6 | 79.6 | 55.2 |
| Spearman | — | — | — |
| nDCG | — | — | — |
| Precision | — | — | — |
| RMSE | — | — | — |

Gene prior R² = 0.32，IMC interaction R² = 0.054（double residual），IMC + prior 可解释总方差的 ~1.6%。

### 1.2 核心瓶颈

#### 瓶颈 #1：冷基因排名几乎与细胞无关

冷基因 g 在细胞 c 中的预测值为：

```
ŷ(c,g) = Φ(g) + β̂_c + x_c^T W H^T y_g
```

在单个细胞内，β̂_c 是常数。IMC 项 `x_c^T W H^T y_g` 的方差仅占总标签方差的 ~1.6%。这意味着在任一细胞中，**冷基因的排名 ~98% 由 Φ(g)（细胞不变）决定**，导致跨细胞的冷基因排名几乎完全相同。

**后果**：计分指标 85% 依赖 per-cell 排名（Spearman 30% + nDCG 30% + Precision@K 25%），冷基因排名 cell-invariant → Spearman → 0 → Cold S → 55。

#### 瓶颈 #2：Gene Prior Φ(g) 信号弱

- 输入：~200 gene features (G1 43 + G2 8 + PW140 140 + coexpr 2 + desc 7)
- 模型：Ridge
- R² = 0.32（仅解释 32% 的基因均值方差）
- 原因：Ridge 线性模型 + 基因特征表达能力有限

#### 瓶颈 #3：IMC 过于无结构

IMC 将所有细胞特征 × 基因特征进行双线性组合（(115+50)×12 ≈ 1,980 参数），但没有编码任何生物学先验。真正有意义的交互（module × indicator, z_cg × evidence）被淹没在大量无意义特征交互中。

---
## 二、文献支撑

### 2.1 核心发现

| 来源 | 关键洞察 | 行动 |
|------|---------|------|
| **Chang & Zhang (2023)** | Multi-output Ridge 预测整个基因依赖谱 ρ=0.88，超越 DeepDEP | Phase 2 |
| **EssSubgraph (2025)** | 表达特征对冷基因泛化最重要，超越网络结构 | 增强 G2 |
| **Macau — BPMF (2018)** | 侧信息作为 latent factor 先验，天然冷启动 | Phase 1 结构设计 |
| **COSINE (2016)** | 加权 profile aggregation — 冷实体的 latent profile 是已知实体 profile 的相似度加权和 | Phase 3 |
| **Deng et al. (2011)** | 仅需 ~2-4% 基因标签即可接近最优 | Phase 4 semi-supervised |
| **DepMap cds-ensemble** | Screen confounders（质量指标）作为 covariate | Phase 4 |
| **EMF (2020)** | 简单 MF + bias 超越 DCell 深度学习模型 | 架构验证 |
| **DREAM challenge** | Multi-task 联合建模优于 per-gene 模型 | Phase 2 |

### 2.2 关键教训

- **不要用深度学习**：Chang & Zhang 证明线性多输出 Ridge 超越深度模型。DeepDEP 的深度堆栈（FCN+AE）被简单 Ridge 击败（ρ=0.88 vs 0.87）。
- **结构化交互 > 无结构交互**：Pint（全成对交互 Lasso）在模拟数据 R²=0.99，在真实 fitness 数据上失败。生物交互不是加性+任意成对，而是受通路/模块约束的。
- **侧信息是关键**：所有成功的冷启动方法（Macau, EMF, COSINE, NG-MC）的共同点是将基因特征/网络作为 latent factor 的正则化先验，而非直接预测冷基因的 latent factor。

---
## 三、优化方案

### 总体架构

```
ŷ(c,g) = μ̂_g + β̂_c + I_structured(c,g) + I_residual(c,g)
```

### Phase 1：结构化生物交互模型 ★★★★★ 核心

**目标**：S +10~15，将 IMC 的弱交互替换为结构化强交互

**当前问题**：IMC R²=0.054，无生物先验，冷基因交互极弱

**方案**：显式编码已知的生物学交互结构：

```
I_structured(c,g) = 
    Σ_k α_k · cell_indicator_k(c) · gene_module_k(g)           (14 params)
  + Σ_k β_k · cell_indicator_k(c) · gene_module_k(g) · z_cg    (14 params)
  + γ₁ · z_cg · evidence_weight(g)                              (1 param)
  + γ₂ · expr_percentile(c,g) · evidence_weight(g)              (1 param)
  + Σ_l Σ_k δ_{l,k} · lineage_l(c) · gene_module_k(g)          (~29×14 → Ridge)
```

**为什么有效**：

1. **Module × Indicator（14 params）**：直接建模 "细胞对模块 k 的依赖 × 基因属于模块 k"。这是线粒体生物学最核心的交互——如果细胞 c 高度依赖 OXPHOS，则所有 OXPHOS 基因在 c 中更 essential。

2. **Expression-modulated Module（14 params）**：同一模块内，表达越高的基因依赖性越强。MITO_RIBOSOME 模块的基因在核糖体需求高的细胞中表达越高的越重要。

3. **Evidence-weighted Expression（2 params）**：高证据权重的基因（如 MitoCarta 确认的），其表达-依赖性耦合更强。低证据基因的表达波动更多是噪声。

4. **Lineage × Module（Ridge）**：不同谱系对线粒体通路有不同依赖模式。如血液谱系 vs 实体瘤对 OXPHOS 的依赖差异。

**冷基因可用性**：所有特征对冷基因都可用：
- `gene_module_k(g)` 来自 MitoCarta 注释（所有基因都有）
- `cell_indicator_k(c)` 来自 Problem 1（所有细胞都有）
- `z_cg` 来自表达矩阵（所有 (cell,gene) 对都有）
- `evidence_weight(g)` 来自 gene_metadata（所有基因都有）
- `lineage_l(c)` 来自 cell_line_metadata（所有细胞都有）

**实现**：
- 构造交互特征矩阵（~450 列）
- RidgeCV 拟合双残差
- 可解释性：α_k 直接表示"模块 k 的 indicator 每 +1σ，属于模块 k 的基因依赖性增加 α_k"

**预期**：交互 R² 从 0.054 → 0.15~0.25

### Phase 2：增强 Gene Prior Φ(g) ★★★★

**目标**：S +3~5，提升冷基因 baseline 精度

**当前问题**：Φ(g) R²=0.32，200 个特征 + 线性 Ridge 不够

**方案 2A：Multi-Output Gene Profile Predictor**

受 Chang & Zhang (2023) 启发——不预测标量基因均值，而是预测整个 1,140 细胞的基因依赖 profile：

```
对于每个基因 g，构建目标向量 y_g ∈ R^{1140}（该基因在所有细胞中的 label）
训练 multi-output Ridge：gene_features(g) → y_g
冷基因：直接从 gene features 预测完整 profile → 细胞特异性预测！
```

这直接解决了"冷基因预测 cell-invariant"的问题——multi-output 预测给出每个细胞的值。

**方案 2B：Co-Expression KNN Profile Transfer**

对于冷基因 g，利用 co-expression 找到最相似的 warm 基因，转移其依赖 profile：

```
ŷ(c, cold_g) = Σ_{warm_g} w(cold_g, warm_g) · y(c, warm_g) / Σ w
其中 w(cold_g, warm_g) = |corr(expr_cold_g, expr_warm_g)|
```

这是 COSINE 风格的加权 profile aggregation。已有的 `compute_coexpression_knn_features` 提供了基因水平的 co-expression KNN，扩展到 cell 水平即可。

**方案 2C：增强基因特征**

1. **模块内均值/方差**：对每个模块，计算基因在其所属模块内的表达均值和方差
2. **Co-expression 网络中心性**：基因在 co-expression 网络中的 degree/betweenness
3. **进化保守性组合特征**：yeast + rickettsia 同源组合（而非 one-hot）
4. **基因描述 NLP 特征增强**：扩展 description keyword 到更多生物学概念

### Phase 3：Gene-Similarity CF 管线集成 ★★★★

**目标**：S +3~5，将已有的 CF 预测真正用于 formula pipeline

**当前问题**：`build_gene_similarity_cf` 已实现但输出从未进入 `predict_formula`。CF 提供细胞特异性冷基因预测，但被浪费了。

**方案**：

1. 在 `train_formula_models` 中增加 CF 组件：
   ```python
   cf_predictions = build_gene_similarity_cf(
       train_labels, gene_meta, pathway_meta, expression, cold_genes, k=20
   )
   ```

2. 在 `predict_formula` 中集成 CF：
   ```python
   # 对冷基因，使用 Pathway-CF 预测替换纯先验
   for i, g in enumerate(gene_ids):
       if g in cold_genes and (cell_ids[i], g) in cf_predictions:
           cf_val = cf_predictions[(cell_ids[i], g)]
           # Blend: (1-λ)·Φ(g) + λ·CF
           mu_arr[i] = (1 - cf_weight) * mu_arr[i] + cf_weight * cf_val
   ```

3. 增强 CF 质量：
   - 使用 PW140（149 通路）替代 14-module 计算基因相似度
   - 结合 co-expression 相似度（已有）形成多核相似度
   - 对冷基因应用 adaptive CF weight（基于最近邻相似度）

**为什么 CF 有效**：COSINE 论文证明加权 profile aggregation 在冷启动中显著优于纯特征预测。Pathway 相似的基因在相同细胞中往往有相似的依赖模式。

### Phase 4：辅助优化 ★★★

**目标**：S +2~5 累积提升

**4A：Per-Cell 分位数校准**
- 为每个 cell 独立做 quantile calibration
- 确保 predicted distribution 匹配 expected distribution
- 单调变换，零风险对 Spearman/NDCG/Precision

**4B：模块锚点半监督**
- Deng et al. 发现只需 2-4% 基因的标签即可接近最优
- 对每个模块选 1-2 个"锚点基因"（冷基因中 evidence 最强的），用 teacher 预测值作为伪标签
- 在模块内做 label propagation

**4C：残差 IMC 保留**
- 在结构化交互之后，用低秩 IMC 捕获剩余的非结构化信号
- `I_residual = IMC(residual_after_structured)`
- rank 降低（3-5），只捕获结构化交互未能解释的模式

**4D：Cold/Warm 混合排序校准**
- 冷基因预测值可能 scale 不同（偏小/偏大）
- Per-cell 冷/热基因分别做 rank-then-merge
- 确保冷基因不会系统性地被排在底部

---
## 四、实现路线图

| 步骤 | 内容 | 预计提升 | 风险 |
|------|------|---------|------|
| **Step 1** | 实现 Phase 1 结构化交互 | S +10~15 | 低：特征全部可用，线性模型 |
| **Step 2** | 集成 Phase 3 Gene-Similarity CF | S +3~5 | 低：代码已实现，只需集成 |
| **Step 3** | 实现 Phase 2A Multi-Output Profile | S +3~5 | 中：需要构造完整 profile 矩阵 |
| **Step 4** | Phase 4A 分位数校准 | S +1~2 | 极低：简单后处理 |
| **Step 5** | Phase 2B Co-Expression Transfer | S +1~3 | 中：需要 OOF 避免泄露 |
| **Step 6** | Phase 4C 残差 IMC | S +1~2 | 低：改动小 |
| **Step 7** | 全量 1140 cells × 1098 genes 验证 | — | — |

**累计预期**：S 从 67 → 80~87

---
## 五、验证策略

1. **快速迭代**：`python local_validate.py --n-genes 200`（200 genes, ~1min）进行每步验证
2. **全量确认**：`python local_validate.py`（全量, ~5min）进行关键节点验证
3. **提交模拟**：`python run.py --task predict` 生成最终提交文件

---
## 六、关键参考

1. Chang & Zhang (2023): "A critical reanalysis of DeepDEP reveals that a simple multi-output ridge regression predicts gene dependency as accurately as deep learning." *bioRxiv*. — Multi-output Ridge beats deep models.
2. Fan et al. (2020): "EMF: extensible matrix factorization for genetic interaction prediction." *Bioinformatics*. — Kernelized side info for cold start.
3. Zakeri et al. (2018): "Macau: scalable Bayesian multi-relational factorization with side information using MCMC." *Bioinformatics*. — BPMF with side info as latent factor prior.
4. Lim et al. (2016): "COSINE: COld Start INformation-based Engine for drug-target interaction prediction." *Sci Rep*. — Weighted profile aggregation for cold entities.
5. Qi et al. (2015): "NG-MC: Network-Guided Matrix Completion." *PLoS Comput Biol*. — GO/PPI-guided matrix completion.
6. Deng et al. (2011): "Predicting gene essentiality across organisms." — ~2-4% labels sufficient.
7. EssSubgraph (2025): "Inductive graph learning for gene essentiality." — Expression features dominate for cold genes.
8. Dempster et al. (2021): "Chronos." *Genome Biology*. — Hierarchical Gaussian kernel prior.
