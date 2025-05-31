import numpy as np
import cv2

def embed_paper_lab_numpy(
    cover_path: str,
    secret_path: str,
    embed_size: int = 64,
    delta: float = 1.0,
    output_npy_path: str = "stego_lab_float.npy"
) -> None:
    """
    将 Secret 隐写到 Cover 的 Lab 频域，并把完整的浮点 Lab 结果存成 .npy：
      - 不做任何 8-bit 量化，保留所有小数位。
    """
    # 1) 读 cover → Lab(float32)
    cover_bgr = cv2.imread(cover_path, cv2.IMREAD_COLOR)
    if cover_bgr is None:
        raise FileNotFoundError(f"Cover 读取失败：{cover_path}")
    lab_cover = cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L_chan = lab_cover[..., 0]                     # float32 in [0..255]
    a_raw  = lab_cover[..., 1].astype(np.float64)  # float64 in [0..255]
    b_raw  = lab_cover[..., 2].astype(np.float64)

    # 2) 转 signed ([-128..127])
    a_signed = a_raw - 128.0  # float64
    b_signed = b_raw - 128.0

    h, w = a_signed.shape

    # 3) FFT → fftshift → mag/phase
    a_fft_shift = np.fft.fftshift(np.fft.fft2(a_signed))
    mag_a       = np.abs(a_fft_shift)   # float64
    phase_a     = np.angle(a_fft_shift)

    b_fft_shift = np.fft.fftshift(np.fft.fft2(b_signed))
    mag_b       = np.abs(b_fft_shift)
    phase_b     = np.angle(b_fft_shift)

    # 4) 读 secret → 拆 R,G → resize → normalize (0..1)
    secret_bgr = cv2.imread(secret_path, cv2.IMREAD_COLOR)
    if secret_bgr is None:
        raise FileNotFoundError(f"Secret 读取失败：{secret_path}")
    R_small = cv2.resize(secret_bgr[..., 2], (embed_size, embed_size)).astype(np.float64)
    G_small = cv2.resize(secret_bgr[..., 1], (embed_size, embed_size)).astype(np.float64)
    r_norm = R_small / 255.0  # float64 in [0..1]
    g_norm = G_small / 255.0

    # 5) 嵌入
    mid_h, mid_w = h // 2, w // 2
    mag_a[mid_h:mid_h + embed_size, mid_w:mid_w + embed_size] += (r_norm * delta)
    mag_b[mid_h:mid_h + embed_size, mid_w:mid_w + embed_size] += (g_norm * delta)

    # 6) 重建频域
    a_fft_shift_new = mag_a * np.exp(1j * phase_a)
    b_fft_shift_new = mag_b * np.exp(1j * phase_b)

    # 7) ifftshift → ifft2 → 取实部 → clamp([-128..127]) → +128 → (0..255) float32
    a_ifft = np.fft.ifft2(np.fft.ifftshift(a_fft_shift_new)).real
    b_ifft = np.fft.ifft2(np.fft.ifftshift(b_fft_shift_new)).real

    a_clamped = np.clip(a_ifft, -128.0, 127.0).astype(np.float32)
    b_clamped = np.clip(b_ifft, -128.0, 127.0).astype(np.float32)

    a_mod = (a_clamped + 128.0).astype(np.float32)  # float32 in [0..255]
    b_mod = (b_clamped + 128.0).astype(np.float32)

    # 8) 合并成完整浮点 Lab（三通道 float32）
    lab_emb = np.stack((L_chan, a_mod, b_mod), axis=-1).astype(np.float32)

    # 9) 直接 np.save，这样 .npy 会保留所有小数。不要存成 PNG 或 TIFF。
    np.save(output_npy_path, lab_emb)
    print(f"[Embed] 已生成浮点 Lab 数组并储存为：{output_npy_path}")


def visualize_stego_from_npy(
    stego_npy_path: str,
    output_png_path: str = "stego.png"
) -> None:
    """
    正确地把“浮点 Lab (.npy)”转成“可视化的 stego.png”：
      1) np.load 读出 Lab_emb (float32)，此时 Lab_emb[...,0] ∈ [0..255], Lab_emb[...,1..2] ∈ [0..255]
      2) 转成 OpenCV 期待的“浮点 Lab”：L' = L * (100/255), a' = a - 128, b' = b - 128
      3) 用 cv2.cvtColor 转成 float BGR，之后 *255 → clamp → uint8 → 写出 PNG
    """
    # 1) 读 .npy
    lab_emb = np.load(stego_npy_path)   # float32, shape = (H, W, 3)
    if lab_emb is None:
        raise FileNotFoundError(f"无法读取 .npy：{stego_npy_path}")

    # （可选）打印一下范围，验证数值没问题
    print(">>> 原始 lab_emb 范围：")
    print("    L ∈", lab_emb[...,0].min(), "~", lab_emb[...,0].max())
    print("    a ∈", lab_emb[...,1].min(), "~", lab_emb[...,1].max())
    print("    b ∈", lab_emb[...,2].min(), "~", lab_emb[...,2].max())

    # 2) 转成 OpenCV 浮点 Lab
    lab_float = np.empty_like(lab_emb, dtype=np.float32)
    #   L 缩放到 [0,100]
    lab_float[..., 0] = lab_emb[..., 0] * (100.0 / 255.0)
    #   a, b 通道去偏移到 [-128, +127]
    lab_float[..., 1] = lab_emb[..., 1] - 128.0
    lab_float[..., 2] = lab_emb[..., 2] - 128.0

    # （可选）再确认一次范围
    print(">>> 转换后 lab_float 范围（OpenCV 浮点 Lab）：")
    print("    L' ∈", lab_float[...,0].min(), "~", lab_float[...,0].max())
    print("    a' ∈", lab_float[...,1].min(), "~", lab_float[...,1].max())
    print("    b' ∈", lab_float[...,2].min(), "~", lab_float[...,2].max())

    # 3) Lab_float → BGR_float
    bgr_float = cv2.cvtColor(lab_float, cv2.COLOR_Lab2BGR)

    # 因为 OpenCV 把“浮点 BGR”当作 [0,1] 范围输出，所以要放大到 0~255
    bgr_u8 = np.clip(bgr_float * 255.0, 0, 255).astype(np.uint8)

    # 写成 PNG
    success = cv2.imwrite(output_png_path, bgr_u8)
    if not success:
        raise RuntimeError(f"写出 stego.png 失败：{output_png_path}")
    print(f"[Visualize] 已成功生成可视化的 stego.png：{output_png_path}")


def extract_paper_lab_numpy(
    cover_path: str,
    stego_npy_path: str,
    embed_size: int = 64,
    delta: float = 1.0,
    output_secret_path: str = "recovered_secret.png"
) -> None:
    """
    非盲式提取 (NumPy 浮点版)：
      1) 读 cover → Lab → 拆 signed a_c, b_c (float64) → FFT→fftshift→mag_a_c, mag_b_c
      2) 读 stego_lab_float.npy → Lab_emb → 拆 signed a_s, b_s → FFT→fftshift→mag_a_s, mag_b_s
      3) 提取频域中心区域：(mag_a_s − mag_a_c)/delta ×255 → clamp → uint8 = R
                            (mag_b_s − mag_b_c)/delta ×255 → uint8 = G
         B=0 → 合并成 (embed_size×embed_size×3) BGR(uint8) → 存 PNG
    """
    # 1) 读 cover → Lab → 拆 signed a_c, b_c
    cover_bgr = cv2.imread(cover_path, cv2.IMREAD_COLOR)
    if cover_bgr is None:
        raise FileNotFoundError(f"Cover 读取失败：{cover_path}")
    lab_cover = cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    a_c_signed = lab_cover[..., 1].astype(np.float64) - 128.0
    b_c_signed = lab_cover[..., 2].astype(np.float64) - 128.0

    mag_a_c = np.abs(np.fft.fftshift(np.fft.fft2(a_c_signed)))
    mag_b_c = np.abs(np.fft.fftshift(np.fft.fft2(b_c_signed)))

    # 2) 读浮点 Lab .npy → 拆 signed a_s, b_s
    lab_emb = np.load(stego_npy_path)  # float32
    a_s_signed = lab_emb[..., 1].astype(np.float64) - 128.0
    b_s_signed = lab_emb[..., 2].astype(np.float64) - 128.0

    mag_a_s = np.abs(np.fft.fftshift(np.fft.fft2(a_s_signed)))
    mag_b_s = np.abs(np.fft.fftshift(np.fft.fft2(b_s_signed)))

    # 3) 提取中心区域
    h, w, _ = lab_emb.shape
    mid_h, mid_w = h // 2, w // 2

    r_norm = (
        mag_a_s[mid_h:mid_h + embed_size, mid_w:mid_w + embed_size]
        - mag_a_c[mid_h:mid_h + embed_size, mid_w:mid_w + embed_size]
    ) / delta

    g_norm = (
        mag_b_s[mid_h:mid_h + embed_size, mid_w:mid_w + embed_size]
        - mag_b_c[mid_h:mid_h + embed_size, mid_w:mid_w + embed_size]
    ) / delta

    R_rec = np.clip(r_norm * 255.0, 0, 255).astype(np.uint8)
    G_rec = np.clip(g_norm * 255.0, 0, 255).astype(np.uint8)
    B_rec = np.zeros_like(R_rec, dtype=np.uint8)

    recovered = cv2.merge((B_rec, G_rec, R_rec))
    cv2.imwrite(output_secret_path, recovered)
    print(f"[Extract] 已成功还原秘密：{output_secret_path}")


if __name__ == "__main__":
    # ============== 请改成你本机上的绝对路径 ==============
    cover_path     = r"cover.png"
    secret_path    = r"secret.png"
    stego_npy_path = r"stego_lab_float.npy"
    stego_png_path = r"stego.png"
    recovered_path = r"recovered_secret.png"
    # =====================================================

    # 1) 嵌入：生成浮点 Lab 并存成 .npy
    embed_paper_lab_numpy(
        cover_path=cover_path,
        secret_path=secret_path,
        embed_size=64,
        delta=1.0,
        output_npy_path=stego_npy_path
    )

    # 2) 可视化成 stego.png（必须把浮点 BGR *255 再 clamp→uint8）
    visualize_stego_from_npy(
        stego_npy_path=stego_npy_path,
        output_png_path=stego_png_path
    )

    # 3) 提取：从浮点 .npy 里还原 R,G 通道的秘密
    extract_paper_lab_numpy(
        cover_path=cover_path,
        stego_npy_path=stego_npy_path,
        embed_size=64,
        delta=1.0,
        output_secret_path=recovered_path
    )

    # （可选）显示还原结果
    import matplotlib.pyplot as plt
    img = cv2.imread(recovered_path)
    plt.figure(figsize=(3,3))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Recovered Rabbit (Full NumPy Workflow)")
    plt.show()
