# IWT+QIM 影像文字隱寫

使用整數小波轉換（Integer Wavelet Transform, IWT）與量化索引調變（Quantization Index Modulation, QIM），實作將 UTF-8 編碼文字嵌入灰階影像子頻帶（LL、LH、HL、HH），並能從水印影像中提取文字。程式採用 Python 撰寫，使用 OpenCV、NumPy、SciPy（FFT via fftpack）等套件。

## 程式說明

檔案 `steg_iwt_qim.py` 內實作了兩個主要功能：

1. `embed_text_iwt_subband(cover_img, text, band, delta)`

   - **功能**：將指定文字（UTF-8）嵌入到 N×M 的灰階 cover 影像中的某一 IWT 子頻帶，輸出水印後影像。
   - **步驟概述**：
     1. 以 `iwt2_haar_integer(img)` 執行一次二維 Haar 整數小波分解，取得四個子頻帶 `(LL, LH, HL, HH)`，每個子頻帶大小為原影像的 1/2。
     2. 將要嵌入的文字轉成位元序列：  
        - 先用 4 bytes（big-endian）表示文字長度（單位：byte），再把每個 byte 拆成 8 位元，合併成完整位元陣列。
     3. 針對選定的子頻帶（LL、LH、HL 或 HH），呼叫 `qim_embed(coeff, bits)`：  
        - 先將子頻帶展平成一維 `flat`，並確認位元長度不超過係數數量，否則拋出容量不足錯誤。  
        - 逐位元處理：對第 i 個位元 `bit`，令原係數 `s = flat[i]`，計算  
          ```
          m = floor(s / (2·δ))
          if bit == 0:
            s' = 2·m·δ
          else:
            s' = (2·m + 1)·δ
          ```
          以此調整係數值，完成 QIM 嵌入。
     4. 將修改後的子頻帶與其他未改動子頻帶一起，以 `iiwt2_haar_integer(LL_mod, LH_mod, HL_mod, HH_mod)` 做反向 IWT 重建，得到水印影像。

2. `extract_text_iwt_subband(watermarked_img, band, delta)`

   - **功能**：從帶有水印的灰階影像中提取先前嵌入的文字，回傳解出的 UTF-8 字串。
   - **步驟概述**：
     1. 以 `iwt2_haar_integer(img)` 執行二維 Haar 整數小波分解，取得 `(LL_w, LH_w, HL_w, HH_w)`。
     2. 針對指定子頻帶取出一維係數 `coeff = 子頻帶.astype(int32).flatten()`。
     3. 逐個係數計算 `r = s % (2·δ)`：若 `r < δ` 則對應位元為 `0`，否則為 `1`，取得完整位元序列 `bits`。
     4. 前 32 位元用以還原文字長度（4 bytes big-endian），計算目標文字總位元數 `total_text_bits = length × 8`。
     5. 取出後續 `total_text_bits` 個位元，分 8 為一組還原成 byte 陣列，再以 UTF-8 解碼（忽略錯誤）成文字。



## 使用方式

```bash
usage: steg_iwt_qim.py [-h] {embed,extract} ...
````

### 1. 嵌入文字（embed）

範例指令：

```bash
python steg_iwt_qim.py embed \
  --input cover.png \
  --output watermarked.png \
  --text "這是一段測試文字" \
  --band LL \
  --delta 1
```

> * `--input,  -i`：輸入灰階 cover 影像路徑（PNG、JPG、BMP 等）。
> * `--output, -o`：輸出水印後影像路徑。
> * `--text,   -t`：要嵌入的 UTF-8 文字內容。若文字過長超過子頻帶容量（子頻帶像素數 ≥ 總位元數），程式會拋出錯誤訊息。
> * `--band,   -b`：選擇要嵌入的子頻帶，可選值爲 `LL`、`LH`、`HL`、`HH`（預設 `LL`）。
> * `--delta,  -d`：QIM 量化間隔 δ，δ 越大則嵌入強度越高，但視覺失真也越明顯；預設為 `1`。

執行成功後，終端會顯示：

```
寫入：watermarked.png
```

### 2. 提取文字（extract）

範例指令：

```bash
python steg_iwt_qim.py extract \
  --input watermarked.png \
  --band LL \
  --delta 1
```

> * `--input,  -i`：欲提取文字的灰階水印影像路徑（必須與 embed 時使用相同檔案與格式）。
> * `--band,   -b`：要提取的子頻帶（需與 embed 時相同）。
> * `--delta,  -d`：QIM 量化間隔 δ（需與 embed 時相同）。

執行後，終端會顯示：

```
提取到的文字：這是一段測試文字
```



## 參考程式結構

```
.
├── README.md
├── steg_iwt_qim.py       # 主程式，包含所有嵌入與提取函式
└── samples/              # （可選）範例影像放置目錄
    ├── cover.png
    └── watermarked.png
```

* `iwt2_haar_integer(img: np.ndarray) -> (LL, LH, HL, HH)`

  * 輸入灰階影像陣列（需偶數高、寬），回傳四個子頻帶之整數陣列（`dtype=int32`）。
* `iiwt2_haar_integer(LL, LH, HL, HH) -> img`

  * 輸入四個子頻帶整數陣列，重建並回傳灰階影像陣列（`dtype=uint8`，範圍 \[0,255]）。
* `embed_text_iwt_subband(cover_img, text, band, delta) -> watermarked_img`

  * 呼叫 IWT 分解，執行 QIM 嵌入，最後 IWT 重建，回傳加水印後的影像陣列。
* `extract_text_iwt_subband(watermarked_img, band, delta) -> text`

  * 呼叫 IWT 分解，解析子頻帶係數取出位元，還原文字長度並解碼，回傳 UTF-8 字串。



## 參考文獻

* Ganic, E., & Eskicioglu, A. M. (2004). Robust DWT‐SVD Domain Image Watermarking: Embedding Data in All Frequencies. MM&SEC '04, 215–219.
