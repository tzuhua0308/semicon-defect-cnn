# WM-811K Wafer Defect Classification

PyTorch CNN pipeline for classifying 9 wafer defect patterns from the WM-811K dataset. v1 baseline reaches **Macro F1 0.734**; v2 adds augmentation + balanced class weights and nearly doubles the hardest class (Scratch F1 **0.192 → 0.351**) — macro F1 drops to 0.688, but in a yield-inspection setting missing a Scratch is far more expensive than a false alarm on `none`, so **v2 is the model chosen for production framing**.

> Sister project: [SECOM Yield Analysis](https://github.com/tzuhua0308/semicon-yield-dashboard) — same domain (semiconductor manufacturing), complementary technique (tabular ML + SQL/Dashboard).
>
> 中文版：[README.md](README.md)

---

## Problem

Wafer maps produced by AOI (Automated Optical Inspection) contain spatial defect patterns that hint at root causes on the fab floor:

| Pattern | Likely root cause |
|---|---|
| Edge-Ring | Wafer edge contact / handling |
| Center | Chuck / centering issue |
| Scratch | Physical damage during transport |
| Donut | Ring-shaped process anomaly |
| Random | Environmental particle contamination |

Automating this classification lets **yield engineers skip visual inspection and jump straight to fixing the machine**.

---

## Dataset

- **Source:** [WM-811K on Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) (`LSWMD.pkl`)
- **Scale:** 811,457 wafers total → **172,950 labeled** (rest unlabeled)
- **Classes:** 9 (`none`, `Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Random`, `Scratch`, `Near-full`)
- **Challenge — extreme class imbalance:**

```
none         147,431  (85.24%)  ██████████████████████████████████████████
Edge-Ring      9,680  ( 5.60%)  ██
Edge-Loc       5,189  ( 3.00%)  █
Center         4,294  ( 2.48%)  █
Loc            3,593  ( 2.08%)  █
Scratch        1,193  ( 0.69%)
Random           866  ( 0.50%)
Donut            555  ( 0.32%)
Near-full        149  ( 0.09%)

Imbalance ratio: 989 : 1
```

---

## Pipeline

```
Raw pkl (629 MB)
   │
   ├── Extract labels from nested arrays (failureType [[none]] → "none")
   ├── Filter to 9 valid classes
   ├── Filter tiny wafers (h/w >= 20)
   ├── Downsample "none" to 5,000  →  training pool 30,516
   │
   ├── Resize to 64×64 (cv2 NEAREST — preserves discrete {0,1,2} values)
   ├── Normalize to [0, 1]
   ├── Stratified split  →  train 70% / val 15% / test 15%
   │
   └── PyTorch Dataset + DataLoader (batch=128)
```

**Model — BaselineCNN (94K params):**
```
Input (B, 1, 64, 64)
  → Conv(1→32) + BN + ReLU + MaxPool  →  (B, 32, 32, 32)
  → Conv(32→64) + BN + ReLU + MaxPool  →  (B, 64, 16, 16)
  → Conv(64→128) + BN + ReLU + MaxPool  →  (B, 128, 8, 8)
  → GlobalAvgPool  →  (B, 128)
  → Linear(128, 9)  →  (B, 9)
```

**Training:** Adam lr=1e-3 · CrossEntropyLoss · 10 epochs · Kaggle GPU · ~5s/epoch

---

## Results

### v1 — Baseline (no augmentation, no class weight)

| Metric | Value |
|---|---|
| Best val accuracy | 0.825 |
| Test macro F1 | **0.7338** |
| Test weighted F1 | 0.8176 |

**Weakness diagnosis:**
- `Scratch` F1 = **0.192** (precision 1.0 but recall only **10.6%**) — the model knows what Scratch looks like, but rarely dares to predict it. Thin diagonal patterns lose most of their signal after 64×64 resize.
- `Donut` recall = 47% (only 555 total samples; the model rarely sees it)
- `Loc` precision = 0.59 (the majority-class loss signal dilutes fine boundary learning)

### v2 — Augmentation + Balanced Weights (**rare-class winner**)

**Techniques:**
- **Augmentation:** `RandomHorizontalFlip` + `RandomVerticalFlip` + `RandomRotation(180°, NEAREST interp)`
- **Class weights:** `sklearn.compute_class_weight('balanced')` → `CrossEntropyLoss(weight=...)` (Near-full 22.8x, Donut 6.1x, Scratch 2.8x)

| Metric | v1 | v2 | Δ |
|---|---|---|---|
| Test macro F1 | 0.7338 | 0.6876 | -0.046 |
| Test weighted F1 | 0.8176 | 0.7489 | -0.069 |
| **Scratch F1** | 0.192 | **0.351** | **+0.159 ↑** |
| **Scratch recall** | 0.106 | **0.274** | **+0.168 ↑** |
| **Donut recall** | 0.470 | **0.723** | **+0.253 ↑** |
| Near-full recall | 0.955 | 0.955 | 0 |
| Random recall | 0.808 | 0.923 | +0.115 ↑ |

**Why v2 is the model production would actually ship:**

In a fab yield setting, **missing a Scratch (false negative) is far more expensive than mislabeling a `none` as something rare (false positive)** — the former lets a defective wafer flow downstream; the latter just triggers one extra visual review. The business-driven metric is rare-class recall, not macro F1. v2 trades 5pp of macro F1 for a nearly 2× Scratch F1 and +25pp Donut recall — the right trade-off in this domain.

### v3 — sqrt-smoothed weights (**failed experiment, kept to show iteration**)

**Hypothesis:** v2's weights are too extreme (Near-full 22.8x). Applying `sqrt` compression to 4.8x should preserve the rare-class boost without crushing the majority classes.

**Result:** Macro F1 0.6744 (worse than v2), Scratch F1 collapsed to **0.115**.

**Diagnosis:** `sqrt` compresses the top correctly, but it also flattens Scratch (2.84 → 1.69) and Random (3.92 → 1.98) — bringing their weights close to the majority classes, so the model stops learning them again. **Lesson:** don't apply `sqrt` uniformly to all classes; compress only the top-k, or switch to focal loss to let the model auto-focus on hard examples.

---

## How to Run

### Kaggle (recommended — free GPU, dataset pre-mounted)
1. Fork the notebook: [Kaggle link — fill in]
2. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2)
3. Run All

### Local
1. Download [`LSWMD.pkl`](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
2. Replace `/kaggle/input/datasets/qingyi/wm811k-wafer-map/LSWMD.pkl` with local path
3. Replace `/kaggle/working/` with a local output directory
4. `pip install -r requirements.txt`
5. GPU strongly recommended (10 epochs ≈ 50s on T4, hours on CPU)

---

## Tech Stack

`Python 3` · `PyTorch` · `torchvision` · `scikit-learn` · `pandas` · `numpy` · `opencv-python` · `matplotlib` · `seaborn`

---

## Design Decisions (for interview talking points)

- **`cv2.INTER_NEAREST` not bilinear** — wafer maps are categorical `{0=empty, 1=pass die, 2=fail die}`; smoothing would create meaningless intermediate values.
- **Downsample `none` to 5,000** — keeps the majority class as a reference but stops it from drowning the loss signal. Alternative was `class_weight='balanced'` alone; downsampling + weighting was chosen for training speed.
- **GAP over Flatten+Dense** — 94K params instead of ~500K, less overfitting on rare classes.
- **Stratified split, not random** — with Near-full (149 samples) a naive split could leave the class out of val or test entirely.

---

## Roadmap

- [x] v1 baseline CNN (Macro F1 0.734)
- [x] v2 augmentation + balanced weights (rare-class winner: Scratch F1 +82%)
- [x] v3 sqrt-weighted retry (failed: 0.674, documented at end of notebook)
- [ ] v4 candidate: focal loss (γ=2) — dynamic weighting of hard examples, best shot at truly beating v1 on macro F1
- [ ] v4 candidate: 96×96 or 128×128 resize — root-cause fix for Scratch's thin-diagonal feature loss
- [ ] SHAP / Grad-CAM visualization (which pixels drove each prediction)
- [ ] Streamlit demo — upload a wafer image → predict pattern
