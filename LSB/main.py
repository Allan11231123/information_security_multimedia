import cv2
import numpy as np
import random
import string
from skimage.metrics import structural_similarity as compare_ssim

def _msg_to_bits(msg: str) -> list:
    """Convert a string to a list of bits, appending sentinel '###END###'."""
    sentinel = "###END###"
    full_msg = msg + sentinel
    bits = []
    for ch in full_msg:
        byte = format(ord(ch), '08b')
        bits.extend([int(b) for b in byte])
    return bits

def calculate_psnr(img1, img2) -> float:
    """Compute PSNR between two 8-bit grayscale images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def encode_lsb(cover_img, msg_bits, k):
    """Embed msg_bits into the k least significant bits of cover_img."""
    h, w = cover_img.shape
    total_pixels = h * w
    capacity = total_pixels * k
    if len(msg_bits) > capacity:
        raise ValueError(f"Message bits ({len(msg_bits)}) exceed capacity ({capacity}).")
    # Pad bits so length is a multiple of k
    pad_len = (k - (len(msg_bits) % k)) % k
    if pad_len:
        msg_bits.extend([0] * pad_len)
    flat = cover_img.flatten()
    num_pixels = len(msg_bits) // k
    for i in range(num_pixels):
        chunk = msg_bits[i*k:(i+1)*k]
        val = 0
        for b in chunk:
            val = (val << 1) | b
        mask = 0xFF ^ ((1 << k) - 1)
        flat[i] = (flat[i] & mask) | val
    return flat.reshape((h, w))

def generate_random_message(num_bytes: int) -> str:
    """Generate a random ASCII message of length num_bytes."""
    choices = string.ascii_letters + string.digits
    return ''.join(random.choices(choices, k=num_bytes))

if __name__ == "__main__":
    cover_path = "lena.png"
    cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    if cover_img is None:
        raise FileNotFoundError("Cannot read lena.png")
    h, w = cover_img.shape
    total_pixels = h * w

    sizes_kb = [1, 4, 8, 16, 32, 64, 128]
    ks = [1, 2, 3]
    results = []

    for k in ks:
        for size in sizes_kb:
            num_bytes = size * 1024
            message = generate_random_message(num_bytes)
            bits = _msg_to_bits(message)
            try:
                stego_img = encode_lsb(cover_img.copy(), bits, k)
            except ValueError:
                # Skip if capacity is insufficient for this k and size
                continue
            cv2.imwrite(f"stego_{size}KB_k{k}.png", stego_img)

            psnr_val = calculate_psnr(cover_img, stego_img)
            ssim_val, _ = compare_ssim(cover_img, stego_img, full=True)
            results.append((size, k, psnr_val, ssim_val))

    print("Size (KB) | k | PSNR (dB) | SSIM")
    print("-------------------------------------")
    for size, k, psnr_val, ssim_val in results:
        print(f"{size:>8} | {k:>1} | {psnr_val:>8.2f} | {ssim_val:.4f}")
