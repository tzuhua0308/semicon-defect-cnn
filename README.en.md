# WM-811K Wafer Defect Classification

PyTorch CNN pipeline for classifying 9 wafer defect patterns from the WM-811K dataset. Baseline model achieves **Macro F1 0.717** on 30K-wafer test split, with a v2 iteration targeting rare-class recall (Scratch, Donut) via augmentation and class weighting.

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

### v1 — Baseline

| Metric | Value |
|---|---|
| Best val accuracy | **0.8018** |
| Test macro F1 | **0.717** |
| Test weighted F1 | (fill in) |

**Weakness diagnosis:**
- `Scratch` recall: **1.7%** — thin diagonal patterns get smeared by 64×64 resize
- `Donut` recall: **42%** — only 555 total samples, model rarely sees it

![v1 confusion matrix](docs/confusion_matrix_v1.png)

### v2 — Augmentation + Class Weights

Techniques applied:
- **Data augmentation:** RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(180°, NEAREST interpolation)
- **Class weighting:** `sklearn.compute_class_weight('balanced')` → `CrossEntropyLoss(weight=...)`

| Class | v1 F1 | v2 F1 | Δ |
|---|---|---|---|
| _(fill in from notebook rerun)_ | | | |

![v2 confusion matrix](docs/confusion_matrix_v2.png)

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

- [x] v1 baseline CNN
- [x] v2 augmentation + class weights (code)
- [ ] v2 rerun + results
- [ ] SHAP / Grad-CAM visualization (which pixels drove each prediction)
- [ ] Try higher resolution (96×96 or 128×128) to rescue Scratch recall
- [ ] Streamlit demo — upload wafer image → predict pattern
