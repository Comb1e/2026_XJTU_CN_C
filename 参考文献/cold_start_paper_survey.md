# 冷启动基因依赖性预测 — 文献调研

调研时间：2026-08-03

---

## 1. 基因依赖性冷启动预测 (CRISPR)

### GATDep (2025)
- **来源**：https://link.springer.com/article/10.1186/s12967-025-07501-3
- **方法**：图注意力网络，整合转录组 (GSVA pathway) 特征与 PPI/GO/KEGG 网络；将依赖性预测建模为基因交互图上的节点回归
- **关键发现**：显式动机为冷启动——大多数肿瘤/罕见癌症无功能筛选数据。批评先前方法"无法整合基因网络等生物先验，限制了冷启动泛化能力"
- **适用性**：GNN 不适用于 sklearn-only 环境，但其设计理念（基因网络先验 → 冷启动）可直接借鉴

### Transcriptome-based Virtual CRISPR Screening (Sadagopan et al., 2024)
- **来源**：https://ouci.dntb.gov.ua/en/works/lRMEMPG4/
- **方法**：从肿瘤转录组推断依赖性，应用于 509 个未筛选细胞系和罕见癌症
- **关键发现**：恢复已知依赖性并发现新脆弱性（包括 TFE3 融合癌的 OXPHOS 依赖 — 直接与线粒体主题相关）
- **适用性**：转录组 → 依赖性的映射思路可直接用于我们的 G2 特征

### DeepDEP (Chiu et al., *Sci Adv* 2021)
- **来源**：https://www.science.org/doi/10.1126/sciadv.abh1275
- **方法**：深度模型，从 (细胞组学, 基因"功能指纹") 预测依赖性。基因侧输入编码为 3,115 维 CGP 特征向量
- **关键发现**：基因功能指纹（CGP signature）是关键的基因表示

### DeepDEP 批判性再分析 (Chang & Zhang, 2023) ★★★★★ 最重要
- **来源**：https://www.biorxiv.org/content/10.1101/2023.11.29.569083v1.full.pdf
- **方法**：独立再分析 DeepDEP，用简单多输出 Ridge 回归预测完整依赖谱
- **关键发现**：
  - **Multi-output Ridge ρ=0.88，超越 DeepDEP (ρ=0.87)**
  - Per-gene ρ=0.276 (Ridge) vs 0.137 (DeepDEP)
  - **结论：精心调优的线性多输出模型是极具竞争力的冷启动 baseline，不需要深度学习**
- **适用性**：★★★★★ 直接指导 Phase 2A 方案

---

## 2. 未见基因泛化

### EssSubgraph (2025) ★★★★★
- **来源**：https://par.nsf.gov/biblio/10672017/media/xml
- **方法**：Inductive GraphSAGE over PPI + expression，最系统的未见基因研究
- **关键发现**：
  - 删除测试基因的标签和特征：AUPRC 0.79 → 0.66–0.67
  - **表达特征对冷基因泛化的贡献大于网络结构**
  - 跨物种 (human→mouse) AUROC 0.79，表明特征可跨基因空间迁移
- **适用性**：★★★★★ 指导 G2（基因表达 profile）优先级提升

### Deng et al. 跨物种基因必要性 (2011) ★★★★★
- **来源**：https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8634067/
- **关键发现**：
  - **仅需 ~2% (原核) / ~4% (真核) 基因标签即可接近最优预测**
  - 组合迁移 + 目标特定学习在标签稀缺时效果最佳
  - 跨物种网络特征可预测必要性
- **适用性**：★★★★★ 指导半监督模块锚点方案；即使极少冷基因标签也能大幅提升

---

## 3. DepMap 数据补全方法

### DepMap "Predictability" Pipeline (cds-ensemble)
- **来源**：https://forum.depmap.org/t/random-forest-code/4261；preprint: 10.1101/2020.02.21.959627
- **方法**：Ensemble (bagged RF/ElasticNet) 从 ~1,000 top-ranked 特征（表达 + 突变 + screen 混杂因素）预测每个基因的依赖性
- **关键发现**：
  - **Screen 质量指标（ScreenMADNonessentials）作为 covariate 显著提升预测**
  - Permutation-based CV 评估泛化能力
- **适用性**：指导添加 per-gene/per-screen 质量特征

### Sagittarius (2022)
- **来源**：https://www.biorxiv.org/content/10.1101/2022.12.24.521845.full.pdf
- **方法**：扰动/药物数据补全
- **关键发现**：**用模型补全数据训练可改善下游必要基因预测**——支持模型补全标签增强训练的策略

### MOSA/MOVE (2024)
- **来源**：https://researchportal.ulisboa.pt/en/publications/synthetic-augmentation-of-cancer-cell-line-multi-omic-datasets-us/
- **方法**：VAE 合成增强 DepMap 多组学数据
- **结论**：验证了合成数据增强对依赖性学习的价值（非 sklearn 级别，仅作参考）

---

## 4. 生物冷启动协同过滤

### COSINE (Lim et al., *Sci Rep* 2016) ★★★★★
- **来源**：https://preview-www.nature.com/articles/srep38860
- **方法**：单类协同过滤 + 低秩矩阵分解用于药物-靶点交互，**显式冷启动求解器：修正加权 profile 方法——新实体的 latent profile 计算为已知实体 profile 的相似度加权聚合**
- **关键发现**：超越 KBMF2K、CMF、NRLMF 等方法（AUPR/AUC）
- **公式**：μ̂_g(cold) = Σ similarity(cold_g, warm_g) × profile(warm_g) / Σ similarity
- **适用性**：★★★★★ 直接指导 Gene-Similarity CF 方案（Phase 3）

---

## 5. 带侧信息的矩阵分解

### EMF — Extensible Matrix Factorization (Fan et al., *Bioinformatics* 2020) ★★★★★
- **来源**：https://academic.oup.com/bioinformatics/article/36/Supplement_2/i866/6055925
- **代码**：https://github.com/lrgr/EMF
- **方法**：可组合的 MF 组件：核化侧信息（从 PPI 网络构建 regularized-Laplacian kernel，将相似基因的 latent factor 拉近）、per-gene bias、跨物种链接
- **关键发现**：
  - 侧信息 + bias 在数据稀疏场景下（10% 训练数据）帮助最大
  - **简单 MF+bias 超越 SOTA 深度模型 DCell，用时 < 1 分钟**
  - 核化侧信息——用基因相似性网络（GO/PPI）正则化 latent factor，而非直接 ridge 预测
- **适用性**：★★★★★ 直接指导基因先验增强方案

### Warped Matrix Factorisation (Pratanwanich et al., 2016)
- **来源**：https://link.springer.com/chapter/10.1007/978-3-319-46227-1_49
- **方法**：侧信息以**低秩协方差先验**形式进入 MF（本体衍生的疾病相似度、表达衍生的基因相似度）
- **适用性**：与 EMF 同族理念，互补实现

---

## 6. 基因必要性迁移学习

### Bingo (Ma et al., *Brief Bioinform* 2023)
- **来源**：https://pmc.ncbi.nlm.nih.gov/articles/PMC10753293/
- **方法**：LLM (ESM-2) + GNN 从蛋白序列做 zero-shot 必要性预测
- **关键发现**：蛋白序列特征携带冷基因信号。sklearn 级别等效：添加蛋白序列描述符（长度、domain、必要性同源物）到 G1
- **适用性**：参考——我们可扩展 G1 中 description 特征

---

## 7. 多任务学习

### DREAM Challenge / Macau on Dependency Data (2018) ★★★★
- **来源**：https://pmc.ncbi.nlm.nih.gov/articles/instance/6022676/bin/bty289_zakeri.129.sup.1.pdf.pdf
- **关键发现**：
  - **联合建模多个基因依赖性（multi-task）优于 per-gene 模型**
  - 共享结构跨相似过程基因传递
  - Macau（贝叶斯 MF）达到最好效果
  - sklearn 等效：Multi-output Ridge/PLSR
- **适用性**：★★★★ 验证 Multi-output Profile Predictor 方向

### CRISPR Multi-Task Analog (2023)
- **来源**：https://www.mdpi.com/2218-273X/13/4/641
- **适用性**：CRISPR 多任务学习的附加验证

---

## 8. 贝叶斯矩阵分解

### Macau — BPMF with Side Information (Zakeri et al., *Bioinformatics* 2018) ★★★★★
- **来源**：https://academic.oup.com/bioinformatics/article/34/13/i447/5048973 (DOI: 10.1093/bioinformatics/bty289)
- **代码**：https://github.com/jaak-s/macau
- **方法**：贝叶斯概率矩阵分解——稀疏基因-表型矩阵，**侧信息通过链接矩阵 (β^T x_i) 作为 latent factor 先验均值**，全贝叶斯 Gibbs 采样
- **关键发现**：
  - 侧信息"使得对无已知关联的基因进行非平凡预测成为可能"——即内置冷启动
  - 基因先验均值 = Ridge(gene_features)，不确定性由证据量调节
  - **我们的 teacher-ridge μ̂_g + shrinkage 是其退化特例；Macau 是原则性泛化**
- **适用性**：★★★★★ 验证并指导经验贝叶斯 shrinkage 方案的改进

---

## 9. 基因本体引导矩阵补全

### NG-MC — Network-Guided Matrix Completion (Qi et al., *PLoS Comput Biol* 2015)
- **来源**：https://pmc.ncbi.nlm.nih.gov/articles/PMC4449711/
- **方法**：低秩概率矩阵补全用于 E-MAP 交互数据，**GO 语义相似度 + PPI 网络作为先验；latent profile 在迭代中通过网络传播**
- **关键发现**：可预测未出现在原始 assay 矩阵中的基因——天然冷启动处理
- **适用性**：label propagation over 14-module network，可用 sklearn label propagation / graph Laplacian regularization 实现

### exp2GO
- **来源**：https://ouci.dntb.gov.ua/?backlinks_to=10.1371%2Fjournal.pone.0153006
- **方法**：NMF + 共享字典耦合基因表达和 GO 语义距离矩阵
- **关键发现**：GO 结构（祖先传播、语义距离）可作为 NMF 级别的核/正则化器

---

## 10. 元学习 / Few-Shot

### Meta-TGLink (Genome Biology 2025)
- **来源**：https://pmc.ncbi.nlm.nih.gov/articles/PMC12636225/
- **方法**：MAML 风格的结构增强图元学习用于 GRN 推理
- **关键发现**：强 few-shot 和 zero-shot 增益；未表征 TF 的 baseline 接近随机 → meta-learning 显著提升
- **适用性**：GNN + meta-learning 不适用于 sklearn-only。但**评估协议可借鉴**：按基因分组 CV + 分别报告 zero-shot 性能（我们已做）

### MetaCaDI (2025)
- **来源**：https://ar5iv.labs.arxiv.org/html/2510.22298
- **方法**：贝叶斯元学习因果发现，**闭式解析适应**（避免梯度内循环）
- **关键发现**：廉价的闭式适应（我们的 ridge teacher + shrinkage）捕获了元学习的大部分收益

---

## 综合结论

### 最高价值行动（sklearn-only）

| 优先级 | 行动 | 来源 | 预期提升 |
|--------|------|------|---------|
| ★★★★★ | **结构化生物交互**（Module×Indicator + Expression-modulated） | 所有 MF+侧信息论文 | S+10~15 |
| ★★★★★ | **Gene-Similarity CF 集成**（COSINE 风格加权 profile aggregation） | COSINE, EMF | S+3~5 |
| ★★★★★ | **Multi-Output Ridge Profile Predictor** | Chang & Zhang 2023 | S+3~5 |
| ★★★★ | **增强基因先验**（核化侧信息、co-expression KNN） | EMF, Macau | S+2~4 |
| ★★★ | **Per-cell 分位数校准** | 标准统计方法 | S+1~2 |
| ★★★ | **半监督模块锚点** | Deng et al. 2011 | S+1~3 |

### 不需要的方向
- ❌ GNN/Transformer meta-learning：线性方法已捕获大部分增益
- ❌ 深度学习（DeepDEP 风格）：被简单 Ridge 超越
- ❌ 全成对交互（Pint 风格）：在真实 fitness 数据上失败
- ❌ VAE 数据增强（MOSA/MOVE）：超出 sklearn 范围
