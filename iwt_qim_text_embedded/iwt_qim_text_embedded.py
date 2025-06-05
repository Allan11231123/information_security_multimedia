import argparse
import numpy as np
import cv2
import scipy
from scipy import fftpack
scipy.fft = fftpack

def iwt2_haar_integer(img: np.ndarray):
    h, w = img.shape
    if h % 2 != 0:
        img = img[:-1, :]
        h -= 1
    if w % 2 != 0:
        img = img[:, :-1]
        w -= 1
    even = img[:, 0::2].astype(np.int32)
    odd  = img[:, 1::2].astype(np.int32)
    Lh = (even + odd) >> 1
    Hh = odd - even
    even_Lh = Lh[0::2, :]
    odd_Lh  = Lh[1::2, :]
    LL = (even_Lh + odd_Lh) >> 1
    LH = odd_Lh - even_Lh
    even_Hh = Hh[0::2, :]
    odd_Hh  = Hh[1::2, :]
    HL = (even_Hh + odd_Hh) >> 1
    HH = odd_Hh - even_Hh
    return LL, LH, HL, HH

def iiwt2_haar_integer(LL: np.ndarray, LH: np.ndarray, HL: np.ndarray, HH: np.ndarray):
    n, m = LL.shape
    even_Lh = LL - (LH >> 1)
    odd_Lh  = LH + even_Lh
    even_Hh = HL - (HH >> 1)
    odd_Hh  = HH + even_Hh
    Lh = np.zeros((2*n, m), dtype=np.int32)
    Hh = np.zeros((2*n, m), dtype=np.int32)
    Lh[0::2, :] = even_Lh
    Lh[1::2, :] = odd_Lh
    Hh[0::2, :] = even_Hh
    Hh[1::2, :] = odd_Hh
    img = np.zeros((2*n, 2*m), dtype=np.int32)
    img[:, 0::2] = Lh - (Hh >> 1)
    img[:, 1::2] = Hh + img[:, 0::2]
    return np.clip(img, 0, 255).astype(np.uint8)


def embed_text_iwt_subband(cover_img: np.ndarray, text: str, band: str, delta: int = 1) -> np.ndarray:
    LL, LH, HL, HH = iwt2_haar_integer(cover_img)

    byte_array = text.encode('utf-8')
    length = len(byte_array)
    length_bytes = length.to_bytes(4, 'big')
    bits = []
    for b in length_bytes:
        for i in range(8):
            bits.append((b >> (7 - i)) & 1)
    for byte in byte_array:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)

    def qim_embed(coeff: np.ndarray, bits: list) -> np.ndarray:
        flat = coeff.flatten().astype(np.int32)
        assert len(bits) <= flat.shape[0], "文字太長，超過容量"
        for i, bit in enumerate(bits):
            s = flat[i]
            m = s // (2 * delta)
            if bit == 0:
                flat[i] = 2 * m * delta
            else:
                flat[i] = (2 * m + 1) * delta
        return flat.reshape(coeff.shape)

    if band == 'LL':
        LL_mod = qim_embed(LL, bits)
        LH_mod, HL_mod, HH_mod = LH, HL, HH
    elif band == 'LH':
        LL_mod = LL
        LH_mod = qim_embed(LH, bits)
        HL_mod, HH_mod = HL, HH
    elif band == 'HL':
        LL_mod, LH_mod = LL, LH
        HL_mod = qim_embed(HL, bits)
        HH_mod = HH
    elif band == 'HH':
        LL_mod, LH_mod, HL_mod = LL, LH, HL
        HH_mod = qim_embed(HH, bits)
    else:
        raise ValueError("band 必須是 LL、LH、HL 或 HH")

    return iiwt2_haar_integer(LL_mod, LH_mod, HL_mod, HH_mod)

def extract_text_iwt_subband(watermarked_img: np.ndarray, band: str, delta: int = 1) -> str:
    LL_w, LH_w, HL_w, HH_w = iwt2_haar_integer(watermarked_img)
    if band == 'LL':
        coeff = LL_w.astype(np.int32).flatten()
    elif band == 'LH':
        coeff = LH_w.astype(np.int32).flatten()
    elif band == 'HL':
        coeff = HL_w.astype(np.int32).flatten()
    elif band == 'HH':
        coeff = HH_w.astype(np.int32).flatten()
    else:
        raise ValueError("band 必須是 LL、LH、HL 或 HH")

    bits = []
    for s in coeff:
        r = s % (2 * delta)
        bits.append(0 if r < delta else 1)

    if len(bits) < 32:
        return ""

    length_bits = bits[:32]
    length = 0
    for b in length_bits:
        length = (length << 1) | b

    total_text_bits = length * 8
    start = 32
    end = start + total_text_bits
    if end > len(bits):
        end = len(bits)
    text_bits = bits[start:end]

    byte_list = []
    for i in range(0, len(text_bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(text_bits):
                byte = (byte << 1) | text_bits[i + j]
            else:
                byte = (byte << 1)
        byte_list.append(byte)

    try:
        text = bytes(byte_list).decode('utf-8', errors='ignore')
    except:
        text = ''
    return text


def main():
    parser = argparse.ArgumentParser(
        description="使用 IWT+QIM 在影像子頻帶中嵌入或提取文字。"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True,
                                       help="embed 或 extract")

    # 嵌入子命令
    parser_embed = subparsers.add_parser("embed", help="將文字嵌入到 cover 影像中")
    parser_embed.add_argument(
        "-i", "--input", required=True,
        help="輸入cover 影像的檔案路徑"
    )
    parser_embed.add_argument(
        "-o", "--output", required=True,
        help="寫入嵌入後影像的檔案路徑"
    )
    parser_embed.add_argument(
        "-t", "--text", required=True,
        help="要嵌入的文字內容（UTF-8 編碼）"
    )
    parser_embed.add_argument(
        "-b", "--band", choices=["LL", "LH", "HL", "HH"], default="LL",
        help="選擇要嵌入的子頻帶，預設為 LL"
    )
    parser_embed.add_argument(
        "-d", "--delta", type=int, default=1,
        help="QIM 量化間隔 delta 值，預設為 1"
    )

    parser_extract = subparsers.add_parser("extract", help="從水印影像中提取文字")
    parser_extract.add_argument(
        "-i", "--input", required=True,
        help="已嵌入水印的灰階影像檔案路徑"
    )
    parser_extract.add_argument(
        "-b", "--band", choices=["LL", "LH", "HL", "HH"], default="LL",
        help="選擇要提取的子頻帶，預設為 LL"
    )
    parser_extract.add_argument(
        "-d", "--delta", type=int, default=1,
        help="QIM 量化間隔 delta 值，必須與嵌入時相同，預設為 1"
    )

    args = parser.parse_args()

    if args.mode == "embed":
        cover_img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        if cover_img is None:
            print(f"[Error] 無法讀取輸入影像：{args.input}")
            return

        watermarked = embed_text_iwt_subband(
            cover_img,
            text=args.text,
            band=args.band,
            delta=args.delta
        )

        cv2.imwrite(args.output, watermarked)
        print(f"寫入：{args.output}")

    elif args.mode == "extract":
        wm_img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        if wm_img is None:
            print(f"[Error] 無法讀取輸入影像：{args.input}")
            return

        recovered_text = extract_text_iwt_subband(
            wm_img,
            band=args.band,
            delta=args.delta
        )
        print(f"提取到的文字：{recovered_text}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
