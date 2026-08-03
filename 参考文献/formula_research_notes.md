# Formula-Based Gene Dependency Prediction — Research Notes

## Chronos (Dempster et al., Genome Biology 2021)

**Core formula**: r_cg = R*_cg / R_c − 1
- r_cg: gene fitness effect (fractional change in growth rate)
- R*_cg: growth rate after knockout of gene g in cell c
- R_c: unperturbed growth rate
- 0 = no effect, negative = dependency, positive = growth advantage

**Cell population model**:
N_cj(t) = N_cj(0) × (p_cj · e^(R*_cg·t) + (1−p_cj) · e^(R_c·t))
- p_cj = p_c × p_j (multiplicative knockout probability decomposition)
- p_c: per-cell-line knockout efficacy
- p_j: per-sgRNA efficacy

**Post-processing** (global, NOT per-cell-line):
- Shift: median of nonessential genes = 0
- Scale: median of common essential genes = −1

**Key insight for our formula**: Multiplicative interaction between gene effect and cell growth rate.
For mitochondrial context: Module_essentiality(g) × Cell_module_dependency(c)

## AC-Chronos (DepMap 23Q2+)

- Within-chromosome-arm median normalization per screen
- Best performer for CN bias correction (Vinceti et al., 2024)
- Best at recapitulating known essential/non-essential genes

## Gene Dependency Decomposition (from our data analysis)

- Gene mean alone: explains 73.8% of label variance
- Cell bias: only 2.0% of variance
- Residual ~24%: gene×cell interaction + noise

## Our Formula Architecture

### Core: y(c,g) = μ̂_g + β̂_c + I(c,g)

### Gene Essentiality μ̂_g
Ridge regression on ~60 gene-level features (G1+G2):
- Module membership (14-dim), sub-mito location (5-dim)
- Evidence weight, expression profile stats (8-dim)
- Pathway count, homolog flags, structural features

### Cell Vulnerability β̂_c
Ridge regression on ~60 cell-level features (G3):
- Module indicators (14-dim), derived indices (10-dim)
- Lineage one-hot (~29-dim), pathway PCA (20-dim)

### Gene×Cell Interaction I(c,g) — FOUR complementary formulas:

1. **Module×Indicator** (14-dim): Σ_m η_m · Module_m(g) · Indicator_m(c)
   → "Gene essentiality in module m × Cell dependency on module m"

2. **Expression effect** (4-dim): θ₁·z + θ₂·|z| + θ₃·max(0,z) + θ₄·min(0,z)
   → Asymmetric nonlinear expression→dependency mapping

3. **Module-match** (14-dim): Σ_m ζ_m · Module_m(g) · ExprPercentile(c,g)
   → Expression percentile weighted by module membership

4. **Evidence-weighted** (1-dim): ω · EvidenceWeight(g) · z_cg
   → High-confidence genes have stronger expression-dependency coupling

### Cold-Start: Gene formula μ̂_g uses only gene features (available for all genes)
+ Pathway-similarity KNN transfer for interaction component

### Sequential Training:
1. Fit μ̂_g on per-gene mean labels
2. Fit β̂_c on per-cell mean residuals
3. Fit I_mod, I_expr, I_match, I_ew on double residuals
4. RidgeCV blend of interaction formulas
5. Final: ŷ = μ̂_g + β̂_c + I_blend

### Innovation for Competition:
- 100% formula-based: every term has named biological coefficients
- Multiplicative interaction derived from Chronos insight
- Multi-scale: module-level (14) + expression-level + pathway-level
- Cold-start by design (gene features only for μ̂_g)
- Human-readable formula printout
