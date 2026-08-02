# 赛题数据说明

本数据包用于“线粒体相关基因扰动依赖性预测”任务。参赛者需要根据细胞系表达特征、通路聚合特征、细胞系元数据和基因元数据，预测给定细胞系在扰动某个线粒体相关基因后的依赖性分数。

每个样本由两列共同确定：

| 字段 | 含义 |
|---|---|
| `cell_line_id` | 细胞系 ID。 |
| `perturbation_gene` | 被扰动的线粒体相关基因，使用 gene symbol。 |

目标列为 `label`。`label` 越大，表示该细胞系对该基因越依赖；扰动该基因后，细胞越容易出现生长劣势。`label` 越小，表示依赖性较弱。

## 目录结构

```text
release/
    README.md
    features/
        cell_expression_raw.csv
        cell_expression_zscore.csv
        cell_pathway_scores_raw.csv
        cell_pathway_scores_zscore.csv
    labels/
        gene_dependency.csv
    metadata/
        cell_line_metadata.csv
        gene_metadata.csv
        pathway_metadata.csv
    submission/
        sample_submission_gene.csv
```

## 主要文件

| 文件 | 用途 |
|---|---|
| `labels/gene_dependency.csv` | 公开训练数据，包含 `cell_line_id`、`perturbation_gene` 和真实 `label`。 |
| `submission/sample_submission_gene.csv` | 赛题提交模板，包含需要预测的 `cell_line_id`、`perturbation_gene`，`label` 为空。 |
| `features/cell_expression_raw.csv` | 细胞系的线粒体相关基因表达特征，数值为 TPM logp1。 |
| `features/cell_expression_zscore.csv` | 对表达特征按基因列做 z-score 标准化后的版本。 |
| `features/cell_pathway_scores_raw.csv` | 基于 raw 表达计算的线粒体通路聚合特征。 |
| `features/cell_pathway_scores_zscore.csv` | 基于表达 z-score 计算的线粒体通路聚合特征。 |
| `metadata/cell_line_metadata.csv` | 细胞系元数据。 |
| `metadata/gene_metadata.csv` | 基因元数据，可用 `perturbation_gene = gene_symbol` 连接。 |
| `metadata/pathway_metadata.csv` | 通路元数据，可与 pathway score 特征列名连接。 |

## 使用方式

1. 使用 `cell_line_id` 连接标签、提交模板、细胞系特征和细胞系元数据。
2. 使用 `perturbation_gene` 连接 `metadata/gene_metadata.csv` 中的 `gene_symbol`。
3. 使用 `labels/gene_dependency.csv` 训练模型。
4. 按 `submission/sample_submission_gene.csv` 的行顺序和键列填写预测 `label`。

## 参考标签数据解释

### `labels/gene_dependency.csv`

每一行是一个已知标签的“细胞系-扰动基因”样本。

| 字段 | 含义 |
|---|---|
| `cell_line_id` | 细胞系 ID，可连接所有 feature 文件和 `metadata/cell_line_metadata.csv`。 |
| `perturbation_gene` | 被扰动基因的 gene symbol，可连接 `metadata/gene_metadata.csv` 的 `gene_symbol`。 |
| `label` | 依赖性标签，数值越大表示该细胞系对该基因越依赖。 |

### `submission/sample_submission_gene.csv`

每一行是一个需要参赛者预测的“细胞系-扰动基因”样本。

| 字段 | 含义 |
|---|---|
| `cell_line_id` | 细胞系 ID，提交时不要修改。 |
| `perturbation_gene` | 被扰动基因，提交时不要修改。 |
| `label` | 需要填写的预测值，模板中为空。 |


## 特征文件格式

所有特征文件都是宽表：第一列为 `cell_line_id`，后续每一列是一个线粒体相关基因/通路。每一行对应一个细胞系。

### 表达特征

`features/cell_expression_raw.csv` 和 `features/cell_expression_zscore.csv` 的列结构相同。

| 列 | 含义 |
|---|---|
| `cell_line_id` | 细胞系 ID。 |
| 其他列，例如 `NDUFAF7`、`SDHA`、`COX10` | 线粒体相关基因的 gene symbol。 |

`cell_expression_raw.csv` 中的数值为 DepMap TPM logp1 表达值。`cell_expression_zscore.csv` 中的数值为按每个基因列在全部 release 细胞系上做 z-score 标准化后的表达值：

```text
z = (x - mean_gene) / (std_gene + 1e-8)
```

### 通路聚合特征

`features/cell_pathway_scores_raw.csv` 和 `features/cell_pathway_scores_zscore.csv` 的列结构相同。

| 列 | 含义 |
|---|---|
| `cell_line_id` | 细胞系 ID。 |
| 其他列，例如 `CI_subunits`、`TCA_cycle`、`Calcium_homeostasis` | 线粒体通路名称，可连接 `metadata/pathway_metadata.csv` 的 `pathway_name`。 |

`cell_pathway_scores_raw.csv` 是通路内可用基因 raw 表达值的简单平均。`cell_pathway_scores_zscore.csv` 是通路内可用基因 z-score 表达值的简单平均。只保留至少有 3 个可用表达基因的通路。

## 元数据字段说明

### `metadata/cell_line_metadata.csv`

每一行对应一个细胞系。

| 字段 | 含义 |
|---|---|
| `cell_line_id` | 细胞系 ID，可连接 label、submission 和 feature 文件。 |
| `CellLineName` | 细胞系显示名称。 |
| `StrippedCellLineName` | 标准化后的细胞系名称，通常去除了部分符号或格式差异。 |
| `OncotreeLineage` | OncoTree lineage，大类组织或肿瘤谱系。 |
| `OncotreePrimaryDisease` | OncoTree 原发疾病名称。 |
| `OncotreeSubtype` | OncoTree 疾病亚型。 |
| `CCLEName` | 细胞系在 CCLE 中的标准名称。 |

### `metadata/gene_metadata.csv`

每一行对应一个可用于建模的线粒体相关基因。

| 字段 | 含义 |
|---|---|
| `gene_symbol` | 基因 symbol，可与 `perturbation_gene` 连接。 |
| `human_gene_id` | 人类 Entrez Gene ID。 |
| `ensembl_gene_id` | Ensembl gene ID。 |
| `uniprot_id` | UniProt 蛋白 ID；一个基因可能对应多个条目。 |
| `synonyms` | 基因别名。 |
| `description` | 基因功能描述。 |
| `curated_gene_list` | 是否属于整理后的线粒体相关基因列表。 |
| `curation_evidence` | 该基因被纳入整理列表的证据摘要。 |
| `sub_mito_location` | 亚线粒体定位注释。 |
| `pathways` | 功能通路注释；一个基因可能包含多个通路，用 `|` 分隔。 |
| `targetp_score` | 线粒体靶向肽预测分数。 |
| `mito_domain_score` | 线粒体相关结构域证据分数。 |
| `coexpression_gnf_n50_score` | 共表达证据分数。 |
| `pgc_induction_score` | PGC 诱导相关证据分数。 |
| `yeast_mito_homolog_score` | 酵母线粒体同源证据分数。 |
| `rickettsia_homolog_score` | 立克次体同源证据分数。 |
| `msms_score` | 质谱证据整合分数。 |

### `metadata/pathway_metadata.csv`

每一行对应一个通路特征。

| 字段 | 含义 |
|---|---|
| `pathway_name` | 通路名称，可与 pathway score 特征列名连接。 |
| `description` | 通路描述。 |
| `n_genes` | 该通路在整理后的基因集合中包含的基因数。 |


## 提交格式

提交文件应保持 `sample_submission_gene.csv` 的三列结构：

| 字段 | 含义 |
|---|---|
| `cell_line_id` | 不要修改。 |
| `perturbation_gene` | 不要修改。 |
| `label` | 填写模型预测值。 |

## 本地验证

参赛者在开发过程中可以基于公开训练数据 `labels/gene_dependency.csv` 自行划分训练集和验证集。将验证集真值保存为包含 `cell_line_id`、`perturbation_gene`、`label` 三列的答案文件，并将模型在同一批样本上的预测保存为提交格式文件，列结构可参考 `submission/sample_submission_gene.csv`。

在项目根目录运行评分脚本：

```bash
conda run -n normal python mito_energy_data_builder/scripts/calculate_metric.py path/to/your_submission.csv path/to/your_answer.csv
```

其中第一个参数是模型预测文件，第二个参数是对应验证集答案文件。两个文件的样本行顺序和 `(cell_line_id, perturbation_gene)` 必须保持一致；预测列可命名为 `label` 或 `prediction`。脚本会输出最终总分以及 Spearman、NDCG、Precision 和 RMSE 各项得分。
