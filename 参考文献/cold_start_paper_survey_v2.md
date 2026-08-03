# 冷启动基因依赖性预测 — 文献调研 v2

调研时间：2026-08-04（补充搜索）

---

## 1. 线性模型超越深度学习 — 最新证据 ★★★★★

### Ahlmann-Eltze, Huber & Anders (2024) — 基因扰动预测
- **来源**：https://www.biorxiv.org/content/10.1101/2024.09.16.613342v4
- **方法**：系统 benchmark transformer 基础模型 (scGPT, scFoundation) 和图神经网络 (GEARS) vs. 简单线性 baseline
- **关键发现**：
  - **未见单基因扰动（冷启动）**：PCA 线性模型 (Ŷ = b + GWP^T) 匹敌/超越 scGPT 和 GEARS
  - **未见双基因扰动**：简单加性模型 大幅超越 所有深度学习方法
  - 预训练 embedding 插入线性模型效果优于完整 transformer 模型
  - 结论："预测未执行实验的结果这一目标尚未实现"
- **适用性**：★★★★★ 直接验证线性模型 + 冷启动方向正确性

### Chang & Zhang (2023) — DeepDEP 批判性再分析 ★★★★★
- **来源**：https://www.biorxiv.org/content/10.1101/2023.11.29.569083v1
- **重新确认**：
  - Multi-output Ridge ρ=0.88 vs DeepDEP ρ=0.87（全局）
  - **Per-gene ρ=0.276 vs 0.137（翻倍！）**— 更严格的冷启动评估
  - 仅表达特征的 Ridge 超越使用全部 4 组组学的 DeepDEP
- **适用性**：★★★★★ 直接指导 Multi-Output Profile Predictor

### Rosenski et al. (2023) — 表达→基因必要性
- **来源**：BMC Medical Genomics, https://pubmed.ncbi.nlm.nih.gov/36803845/
- **方法**：线性模型识别小量"修饰基因"预测必要性
- **关键发现**：超越 SOTA 深度模型（可预测基因数 ~3,000 + 准确度），可解释、不过拟合

### Sharma et al. (2025) — DepMap 组学→转移预测
- **来源**：https://www.biorxiv.org/content/10.1101/2025.02.15.638428v1
- **方法**：9 种线性和非线性模型比较（481 细胞系 × 21 肿瘤类型）
- **关键发现**：
  - 线性和非线性模型间 无显著差异 (Mann-Whitney q≥0.86)
  - **样本量而非模型复杂度是限制因素**
  - 线性 SVR 排名第一

---

## 2. 排序-校准解耦方法 ★★★★★ NEW

### CAIRO: Calibrate After Initial Rank Ordering (2026) ★★★★★
- **来源**：https://ar5iv.labs.arxiv.org/html/2602.14440
- **方法**：两阶段框架
  - **Stage 1**：用 scale-invariant 排序损失学习 g(x)（Spearman ρ / Kendall τ / Gini covariance）
    - 这些损失对离群值和重尾噪声鲁棒
    - ORO (Optimal-in-Rank-Order) 性质：最小化器是条件期望的严格单调变换
  - **Stage 2**：Isotonic Regression (PAV 算法) 恢复 scale
    - 定理：f*(g*(X)) = m*(X) 在总体水平恢复真实回归函数
    - 严格 auto-calibration：预测值在每个水平集内等于目标期望
- **适用性**：★★★★★ **核心突破**
  - 评分脚本 85% 权重依赖 per-cell 排名 (Spearman 30% + nDCG 30% + Precision@K 25%)
  - **优化排序优于优化 MSE！**
  - Stage 1 用 Spearman/Kendall 损失训练 → 直接优化评分指标
  - Stage 2 isotonic calibration 确保 RMSE 也不差

### Menon et al. (ICML 2012) — Rank + Isotonic Regression
- **来源**：https://icml.cc/2012/papers/372.pdf
- **关键发现**：Bayes-optimal ranker + calibrated → 恢复真实概率
  - Isotonic regression 保留输入分数的 AUC
  - 平方误差上界 = √(n⁺n⁻)/(2(n⁺+n⁻)) · √(1−AUC)

### NIPS 2013 — Regret Transfer: Ranking → Probability Estimation
- **来源**：https://shivaniagarwal.net/wp-content/uploads/2024/09/nips13-classification-ranking-cpe.pdf
- **关键发现**：任何统计一致的排序算法可通过 isotonic calibration 转换为一致的概率估计
  - 平方误差 regret ≤ √(8p(1-p)·regret_rank) + O((ln n/n)^{1/3})

---

## 3. 冷启动协同过滤与矩阵分解

### COSINE (Lim et al., Sci Rep 2016) ★★★★★
- **来源**：https://www.nature.com/articles/srep38860
- **方法**：加权 profile aggregation — 冷实体 latent profile = Σ similarity × 已知实体 profile
- **公式**：μ̂_g(cold) = Σ sim(cold_g, warm_g) × profile(warm_g) / Σ sim
- **适用性**：★★★★★ 基因相似性 CF 核心参考

### EMF — Extensible Matrix Factorization (Fan et al., Bioinformatics 2020) ★★★★★
- **来源**：https://academic.oup.com/bioinformatics/article/36/Supplement_2/i866/6055925
- **方法**：核化侧信息正则化 latent factor — 相似基因的 latent factor 被拉近
- **关键发现**：
  - 简单 MF+bias 超越 DCell 深度模型（< 1 分钟）
  - 侧信息在数据稀疏场景（10%）帮助最大
- **适用性**：★★★★★ 指导 SVD 因子的图正则化

### Macau — BPMF with Side Information (Zakeri et al., Bioinformatics 2018) ★★★★★
- **来源**：https://academic.oup.com/bioinformatics/article/34/13/i447/5048973
- **方法**：侧信息通过链接矩阵 (β^T x_i) 作为 latent factor 先验均值
- **关键发现**：侧信息"使对无已知关联的基因进行非平凡预测成为可能"
- **适用性**：★★★★★ 验证经验贝叶斯 shrinkage 的贝叶斯基础

### NG-MC — Network-Guided Matrix Completion (Qi et al., PLoS Comput Biol 2015)
- **来源**：https://pmc.ncbi.nlm.nih.gov/articles/PMC4449711/
- **方法**：GO 语义相似度 + PPI 网络作为低秩矩阵补全先验
- **关键发现**：latent profile 在迭代中通过网络传播
- **适用性**：label propagation over MitoCarta 149 pathways

---

## 4. 图正则化与半监督学习

### DA-HGL (2025) — NMF + 双图正则化 ★★★★
- **来源**：https://pmc.ncbi.nlm.nih.gov/articles/PMC12476837/
- **方法**：非负矩阵分解 + 蛋白相似图 Laplacian + GO 语义图 Laplacian
- **关键发现**：显式冷启动场景表现优异 (Fmax +9~17%)
- **适用性**：★★★ 图正则化 SVD/IMC 因子的技术参考

### EssSubgraph (2025) ★★★★★
- **已记录**（v1 文献调研）
- **补充**：跨物种 (human→mouse) AUROC 0.79，证明特征可跨基因空间迁移
- **表达特征对冷基因泛化贡献 > 网络结构**

---

## 5. 综合结论与优化方向

### 核心洞察

1. **优化排序而非 MSE**：评分脚本 85% 来自排序指标。MSE 最小化 ≠ 排序最优。CAIRO 框架提供原则性的排序-校准解耦。

2. **线性模型足够**：4 篇独立论文（Chang & Zhang 2023, Ahlmann-Eltze 2024, Rosenski 2023, Sharma 2025）一致表明线性模型匹敌/超越深度模型。

3. **Multi-output > Scalar**：预测完整依赖 profile 而非标量基因均值，直接解决冷基因 cell-invariant 问题。

4. **图正则化侧信息 > 无结构 MF**：EMF/Macau/NG-MC/DA-HGL 一致表明，通过生物网络正则化 latent factor 优于纯低秩分解。

5. **2-4% 标签足够**：Deng et al. (2011) — 半监督学习的理论基础。

### 优先级排序（sklearn-only 可解释模型）

| 优先级 | 方向 | 核心论文 | 预期 S 提升 | 实现复杂度 |
|--------|------|---------|------------|-----------|
| ★★★★★ | **CAIRO 排序优先训练** | CAIRO 2026 | +5~10 | 中 |
| ★★★★★ | **Multi-Output Profile Predictor** | Chang & Zhang 2023 | +3~5 | 中 |
| ★★★★★ | **图正则化 SVD (Pathway-Laplacian)** | EMF, NG-MC, DA-HGL | +2~4 | 中 |
| ★★★★ | **多核基因相似性 CF** | COSINE, EMF | +2~3 | 低 |
| ★★★★ | **Per-Cell 分位数对齐** | CAIRO, 标准统计 | +2~3 | 低 |
| ★★★ | **增强基因先验特征** | EssSubgraph, Macau | +1~3 | 低 |
| ★★★ | **Label Propagation on Pathways** | Deng 2011, NG-MC | +1~3 | 中 |
