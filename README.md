# WM-811K 晶圓缺陷分類

用 PyTorch CNN 對 WM-811K 資料集的 9 種晶圓缺陷 pattern 做分類。v1 baseline 在 test set 拿到 **Macro F1 0.734**，v2 加入資料擴增 + 平衡權重把最難的 Scratch F1 從 **0.192 → 0.351**（近 2x）——雖然 macro F1 掉到 0.688，但在良率場域裡「漏掉一個 Scratch」比「誤報 none」貴得多，v2 才是實務會採用的模型。

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

### v1 — Baseline（無 augmentation、無 class weight）

| 指標 | 數值 |
|---|---|
| Best val accuracy | 0.825 |
| Test macro F1 | **0.7338** |
| Test weighted F1 | 0.8176 |

**弱點診斷：**
- `Scratch` F1 = **0.192**（precision 1.0、recall 只有 **10.6%**）——模型知道 Scratch 長什麼樣，只是不敢多預測。細長對角 pattern 被 64×64 resize 抹掉大部分特徵。
- `Donut` recall = 47%（總共只有 555 筆樣本，模型看得少）
- `Loc` precision = 0.59（多數類 loss 訊號稀釋了細部邊界學習）

### v2 — 資料擴增 + 平衡權重（**rare-class 贏家**）

**手法：**
- **Augmentation：** `RandomHorizontalFlip` + `RandomVerticalFlip` + `RandomRotation(180°, NEAREST 插值)`
- **Class weight：** `sklearn.compute_class_weight('balanced')` → `CrossEntropyLoss(weight=...)`（Near-full 22.8x、Donut 6.1x、Scratch 2.8x）

| 指標 | v1 | v2 | Δ |
|---|---|---|---|
| Test macro F1 | 0.7338 | 0.6876 | -0.046 |
| Test weighted F1 | 0.8176 | 0.7489 | -0.069 |
| **Scratch F1** | 0.192 | **0.351** | **+0.159 ↑** |
| **Scratch recall** | 0.106 | **0.274** | **+0.168 ↑** |
| **Donut recall** | 0.470 | **0.723** | **+0.253 ↑** |
| Near-full recall | 0.955 | 0.955 | 0 |
| Random recall | 0.808 | 0.923 | +0.115 ↑ |

**為什麼 v2 才是實務會採用的模型？**

在 fab 良率場域，**漏掉一個 Scratch 缺陷（false negative）比錯報一個 none（false positive）貴得多**——前者可能讓不良晶圓進入下一站，後者只是多做一次目視 review。business-driven metric 是稀有類的 recall，不是 macro F1。v2 用 5pp macro F1 換到 Scratch F1 幾乎翻倍、Donut recall +25pp，符合這個 trade-off。

### v3 — 嘗試 sqrt 平滑權重（**失敗實驗，保留以顯示迭代思維**）

**假設：** v2 的權重太極端（Near-full 22.8x），用 `sqrt` 壓縮到 4.8x 應該能保留 rare-class boost 又不會壓垮多數類。

**結果：** Macro F1 0.6744（比 v2 還低）、Scratch F1 崩到 **0.115**。

**診斷：** sqrt 對「頂端」有效沒錯，但把 Scratch（2.84 → 1.69）和 Random（3.92 → 1.98）也一起壓扁。這兩類的權重和多數類差不多後，模型又回到不去救稀有類的行為。**教訓：不該全類一起 sqrt，該只對 top-k 壓縮，或改用 focal loss 讓模型自己找 hard example。**

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

- [x] v1 baseline CNN（Macro F1 0.734）
- [x] v2 augmentation + balanced weights（rare-class 贏家：Scratch F1 +82%）
- [x] v3 sqrt-weighted retry（失敗：0.674，記錄於 notebook 尾）
- [ ] v4 candidate：focal loss（γ=2）——動態加權 hard 樣本，最有機會真正超越 v1 macro F1
- [ ] v4 candidate：96×96 或 128×128 resize——從根本解 Scratch 的細對角特徵遺失
- [ ] SHAP / Grad-CAM 視覺化（哪些 pixel 主導每個預測）
- [ ] Streamlit demo — 上傳晶圓圖 → 預測 pattern
