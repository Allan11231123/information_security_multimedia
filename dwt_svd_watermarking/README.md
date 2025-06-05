# DWT-SVD Image Watermarking

使用Ganic & Eskicioglu (2004)〈Robust DWT‐SVD Domain Image Watermarking: Embedding Data in All Frequencies〉裡DWT結合奇異值分解（SVD）的方法，實現影像水印的嵌入與提取。程式採用 Python 撰寫，使用 PyWavelets、OpenCV、NumPy 等套件。

## 程式說明

檔案 `dwt_svd_watermarking.py` 內實作了兩個功能：

1. `embed_watermark(cover_img, watermark, alpha_ll, alpha_others, wavelet)`

   * **功能**：將 N×N 的灰階水印嵌入到大小為 2N×2N 的灰階封面影像裡
   * **步驟概述**：

     1. 以 PyWavelets 做一次 2D DWT，將封面影像分解為 LL、LH、HL、HH 四個子頻帶
     2. 分別對每個子頻帶做 SVD，取其奇異值向量 S，並以 `alpha_ll`（對 LL）或 `alpha_others`（對 LH/HL/HH）放大後，將原始水印的奇異值乘上縮放係數疊加到 S 的前 N 個元素中
     3. 依照修改後的奇異值與原 U, V<sup>T</sup> 重建每個子頻帶，再做逆 DWT 合成出有水印的影像

2. `extract_watermark(watermarked_img, original_cover, watermark, alpha_ll, alpha_others, wavelet)`

   * **功能**：從嵌入水印後的影像中，並利用原始未嵌水印的封面影像與原始水印影像，提取出水印
   * **步驟概述**：

     1. 對「有水印的影像」與「原始封面影像」各做一次 2D DWT，取得四子頻帶（LL、LH、HL、HH）
     2. 針對相同子頻帶做 SVD，得到兩組奇異值向量 S<sub>w</sub> 與 S<sub>o</sub>，計算 `(S<sub>w</sub> - S<sub>o</sub>) / alpha` 得到經過縮放前的水印奇異值向量估計值 S<sub>est</sub>
     3. 針對水印原始影像（N×N）做一次 SVD，取得水印的左、右奇異向量（U<sub>w</sub>、V<sub>w</sub><sup>T</sup>），再以 U<sub>w</sub> · diag(S<sub>est</sub>) · V<sub>w</sub><sup>T</sup> 重建出水印影像


## 使用方式


```bash
usage: dwt_svd_watermarking.py [-h] {embed,extract} ...
```

### 1. 嵌入水印

範例指令：

```bash
python dwt_svd_watermarking.py embed \
  --cover example_images/cover.png \
  --watermark example_images/watermark.png \
  --output outputs/watermarked.png \
  --alpha_ll 0.05 \
  --alpha_others 0.005 \
  --wavelet haar
```

> * `--cover`：原始灰階封面影像路徑，大小必須為 2N×2N。
> * `--watermark`：原始灰階水印影像路徑，大小必須為 N×N (正方形)。
> * `--output`：嵌入水印後影像的輸出路徑。
> * `--alpha_ll`：LL 頻帶的縮放係數 (預設 0.05)。
> * `--alpha_others`：LH、HL、HH 頻帶的縮放係數 (預設 0.005)。
> * `--wavelet`：使用的小波名稱 (例如 `haar`, `db2` 等)。

執行成功後，會在 `outputs/watermarked.png` 看到加入隱形水印的結果。

### 2. Extract 模式（提取水印）

範例指令：

```bash
python dwt_svd_watermarking.py extract \
  --watermarked outputs/watermarked.png \
  --cover example_images/cover.png \
  --watermark example_images/watermark.png \
  --output outputs/extracted.png \
  --alpha_ll 0.05 \
  --alpha_others 0.005 \
  --wavelet haar
```

> * `--watermarked`：剛剛 embed 完成的「有水印影像」路徑。
> * `--cover`：原始未嵌水印的封面影像路徑。
> * `--watermark`：原始水印影像路徑（用於 SVD 基底重建）。
> * `--output`：提取出來的水印影像輸出路徑。
> * 請與 embed 時使用的值保持一致，才能還原出正確位置與強度的水印。

## 範例演示

假設放在 `example_images/` 中的範例：

* `cover.png`：尺寸 256×256，灰階
* `watermark.png`：尺寸 128×128，灰階（內容任意，可為文字或圖案）

1. 先 embed：

   ```bash
   python dwt_svd_watermarking.py embed \
     --cover example_images/cover.png \
     --watermark example_images/watermark.png \
     --output outputs/watermarked.png
   ```

2. 接著 extract：

   ```bash
   python dwt_svd_watermarking.py extract \
     --watermarked outputs/watermarked.png \
     --cover example_images/cover.png \
     --watermark example_images/watermark.png \
     --output outputs/extracted.png
   ```

## 參考文獻

* Ganic, E., & Eskicioglu, A. M. (2004). Robust DWT‐SVD Domain Image Watermarking: Embedding Data in All Frequencies. *MM\&SEC '04*, 215–219.


