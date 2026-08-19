# WM-811K 晶圓缺陷分類

用 PyTorch CNN 對 WM-811K 資料集的 9 種晶圓缺陷 pattern 做分類。Baseline 模型在 30K 張晶圓的 test set 上達到 **Macro F1 0.717**，v2 版加入資料擴增與類別權重，目標救回少數類（Scratch、Donut）的 recall。

> 姐妹作：[SECOM Yield Analysis](https://github.com/tzuhua0308/semicon-yield-dashboard) — 同樣是半導體製造領域，但用互補的技術（tabular ML + SQL/Dashboard）。
>
> English version: [README.en.md](README.en.md)

---

## 問題定義

AOI（自動光學檢測）機台掃出來的晶圓圖上，缺陷分布的**空間 pattern** 會暗示 fab 內的 root cause：

| Pattern | 可能原因 |
|---|---|
| Edge-Ring | 晶圓邊緣接觸 / 搬運問題 |
| Center | 卡盤 / 對心異常 |
| Scratch | 搬運過程被刮傷 |
| Donut | 環狀製程異常 |
| Random | 環境粒子污染 |

自動化這一步分類，能讓**良率工程師跳過目視檢查，直接去修機台**。

---

## 資料集

- **來源：** [WM-811K on Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)（`LSWMD.pkl`）
- **規模：** 總共 811,457 張晶圓 → **有標籤 172,950 張**（其餘無標籤）
- **類別：** 9 類（`none`、`Center`、`Donut`、`Edge-Loc`、`Edge-Ring`、`Loc`、`Random`、`Scratch`、`Near-full`）
- **難點——極度類別不平衡：**

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

不平衡比例：989 : 1
```

---

## Pipeline

```
原始 pkl（629 MB）
   │
   ├── 從巢狀 array 抽 label（failureType [[none]] → "none"）
   ├── 過濾到 9 個有效類別
   ├── 過濾異常小尺寸（h/w >= 20）
   ├── 把 "none" downsample 到 5,000  →  訓練池 30,516 筆
   │
   ├── Resize 到 64×64（cv2 NEAREST，保留離散值 {0,1,2}）
   ├── Normalize 到 [0, 1]
   ├── Stratified 切分  →  train 70% / val 15% / test 15%
   │
   └── PyTorch Dataset + DataLoader（batch=128）
```

**模型 — BaselineCNN（94K 參數）：**
```
Input (B, 1, 64, 64)
  → Conv(1→32) + BN + ReLU + MaxPool  →  (B, 32, 32, 32)
  → Conv(32→64) + BN + ReLU + MaxPool  →  (B, 64, 16, 16)
  → Conv(64→128) + BN + ReLU + MaxPool  →  (B, 128, 8, 8)
  → GlobalAvgPool  →  (B, 128)
  → Linear(128, 9)  →  (B, 9)
```

**訓練：** Adam lr=1e-3 · CrossEntropyLoss · 10 epochs · Kaggle GPU · 每 epoch 約 5 秒

---

## 結果

### v1 — Baseline

| 指標 | 數值 |
|---|---|
| Best val accuracy | **0.8018** |
| Test macro F1 | **0.717** |
| Test weighted F1 | （待補） |

**弱點診斷：**
- `Scratch` recall：**1.7%** — 細長對角 pattern 被 64×64 resize 抹掉
- `Donut` recall：**42%** — 總共只有 555 筆樣本，模型很少看到

![v1 confusion matrix](docs/confusion_matrix_v1.png)

### v2 — 資料擴增 + 類別權重

採用手法：
- **資料擴增：** RandomHorizontalFlip、RandomVerticalFlip、RandomRotation(180°, NEAREST 插值)
- **類別權重：** `sklearn.compute_class_weight('balanced')` → `CrossEntropyLoss(weight=...)`

| 類別 | v1 F1 | v2 F1 | Δ |
|---|---|---|---|
| _（notebook 重跑後填入）_ | | | |

![v2 confusion matrix](docs/confusion_matrix_v2.png)

---

## 如何執行

### Kaggle（推薦——免費 GPU，資料集已掛載）
1. Fork notebook：[Kaggle 連結——待補]
2. 開啟 GPU 加速（Settings → Accelerator → GPU T4 x2）
3. Run All

### 本地
1. 下載 [`LSWMD.pkl`](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
2. 把 `/kaggle/input/datasets/qingyi/wm811k-wafer-map/LSWMD.pkl` 換成本地路徑
3. 把 `/kaggle/working/` 換成本地輸出路徑
4. `pip install -r requirements.txt`
5. 強烈建議用 GPU（10 epochs 在 T4 上約 50 秒，CPU 要跑數小時）

---

## 技術棧

`Python 3` · `PyTorch` · `torchvision` · `scikit-learn` · `pandas` · `numpy` · `opencv-python` · `matplotlib` · `seaborn`

---

## 技術選型（面試可講的重點）

- **用 `cv2.INTER_NEAREST` 而不是 bilinear** — 晶圓圖是離散類別 `{0=空、1=良品 die、2=不良 die}`，用平滑插值會產生沒有意義的中間值。
- **把 `none` downsample 到 5,000** — 保留多數類當參考，同時不讓它主導 loss 訊號。另一條路是只用 `class_weight='balanced'`，但為了訓練速度選擇 downsample + weighting 併用。
- **用 GAP 取代 Flatten+Dense** — 94K 參數而非 ~500K，少數類的過擬合風險更低。
- **Stratified split 而非隨機切** — Near-full 只有 149 筆，隨機切可能整類都不進 val 或 test。

---

## Roadmap

- [x] v1 baseline CNN
- [x] v2 augmentation + class weights（code 已完成）
- [ ] v2 重跑 + 結果
- [ ] SHAP / Grad-CAM 視覺化（哪些 pixel 主導每個預測）
- [ ] 嘗試更高解析度（96×96 或 128×128）救 Scratch recall
- [ ] Streamlit demo — 上傳晶圓圖 → 預測 pattern
