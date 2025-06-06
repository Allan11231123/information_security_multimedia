import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    計算兩張同尺寸灰階影像的 PSNR (Peak Signal-to-Noise Ratio)。
    - img1, img2: 8-bit 灰階影像 (values 0–255)，需同尺寸。
    回傳 PSNR 值 (單位 dB)。
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')  # 完全相同
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    計算兩張同尺寸灰階影像的 SSIM (Structural Similarity Index)。
    - img1, img2: 8-bit 灰階影像 (values 0–255)，需同尺寸。
    回傳 SSIM 值 (介於 -1 到 1 之間，1 表示完全相同)。
    """
    # compare_ssim 預設會把輸入視作浮點數並歸一化到 [0,1]，所以我們直接傳 0–255 的灰階圖也沒問題
    ssim_value, _ = compare_ssim(img1, img2, full=True)
    return ssim_value

if __name__ == "__main__":
    # 範例：讀同一尺寸的封面圖 & Stego 圖（灰階）
    cover_path = "fb05382ac3713c516b2906e3764f64ae.jpg"   # 你藏完訊息前的灰階封面
    stego_path = "stego.png"   # 你藏完訊息後的 Stego 圖
    secret_path = "secret.png"  # 你的灰階秘密圖 (程式會自動 resize)
    recover_path = "recovered.png"
    # 讀入灰階影像
    cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    stego_img = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
    secret_img = cv2.imread(secret_path, cv2.IMREAD_GRAYSCALE)
    recover_img = cv2.imread(recover_path, cv2.IMREAD_GRAYSCALE)

    h1, w1 = secret_img.shape[:2]
    target_size = (w1, h1)

    resized1 = cv2.resize(secret_img, target_size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(recover_path, resized1)


    if cover_img is None or stego_img is None:
        raise FileNotFoundError("無法讀到其中一張圖片，請確認路徑與檔名是否正確。")

    # 確認尺寸相同
    if cover_img.shape != stego_img.shape:
        raise ValueError("兩張圖片尺寸不同，無法計算 SSIM/PSNR。")

    # 計算 PSNR
    psnr_value = calculate_psnr(cover_img, stego_img)
    print(f"PSNR (cover vs. stego) = {psnr_value:.2f} dB")

    # 計算 SSIM
    ssim_value = calculate_ssim(cover_img, stego_img)
    print(f"SSIM (cover vs. stego) = {ssim_value:.4f}")

    psnr_value = calculate_psnr(secret_img, recover_img)
    print(f"PSNR (secret_img vs. recover_img) = {psnr_value:.2f} dB")

    # 計算 SSIM
    ssim_value = calculate_ssim(secret_img, recover_img)
    print(f"SSIM (secret_img vs. recover_img) = {ssim_value:.4f}")
