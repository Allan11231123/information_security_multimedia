import argparse
import numpy as np
import scipy
from scipy import fftpack
scipy.fft = fftpack

import pywt
import cv2
import os


def embed_watermark(cover_img: np.ndarray, watermark: np.ndarray,
                    alpha_ll: float = 0.05, alpha_others: float = 0.005,
                    wavelet: str = 'haar') -> np.ndarray:
    
    h, w = cover_img.shape
    n, m = watermark.shape
    assert h == w == 2 * n, "Cover image must be twice the size of watermark"
    assert n == m, "Watermark must be square"

    LL, (LH, HL, HH) = pywt.dwt2(cover_img, wavelet)

    def svd_and_embed(subband, alpha):
        U, S, Vt = np.linalg.svd(subband, full_matrices=False)
        _, Sw, _ = np.linalg.svd(watermark, full_matrices=False)
        S_mod = S.copy()
        S_mod[:n] += alpha * Sw
        return U @ np.diag(S_mod) @ Vt

    LL_mod = svd_and_embed(LL, alpha_ll)
    LH_mod = svd_and_embed(LH, alpha_others)
    HL_mod = svd_and_embed(HL, alpha_others)
    HH_mod = svd_and_embed(HH, alpha_others)

    watermarked = pywt.idwt2((LL_mod, (LH_mod, HL_mod, HH_mod)), wavelet)
    return watermarked


def extract_watermark(watermarked_img: np.ndarray, original_cover: np.ndarray,
                      watermark: np.ndarray,
                      alpha_ll: float = 0.05, alpha_others: float = 0.005,
                      wavelet: str = 'haar'):

    LL_w, (LH_w, HL_w, HH_w) = pywt.dwt2(watermarked_img, wavelet)
    LL_o, (LH_o, HL_o, HH_o) = pywt.dwt2(original_cover, wavelet)

    def svd_and_extract(sub_w, sub_o, alpha):
        _, Sw_w, _ = np.linalg.svd(sub_w, full_matrices=False)
        _, So, _ = np.linalg.svd(sub_o, full_matrices=False)
        return (Sw_w - So) / alpha

    S_ll = svd_and_extract(LL_w, LL_o, alpha_ll)
    S_lh = svd_and_extract(LH_w, LH_o, alpha_others)
    S_hl = svd_and_extract(HL_w, HL_o, alpha_others)
    S_hh = svd_and_extract(HH_w, HH_o, alpha_others)


    n = watermark.shape[0]
    Uw, _, Vtw = np.linalg.svd(watermark, full_matrices=False)

    W_ll = Uw[:, :n] @ np.diag(S_ll) @ Vtw[:n, :]
    W_lh = Uw[:, :n] @ np.diag(S_lh) @ Vtw[:n, :]
    W_hl = Uw[:, :n] @ np.diag(S_hl) @ Vtw[:n, :]
    W_hh = Uw[:, :n] @ np.diag(S_hh) @ Vtw[:n, :]

    return {
        'LL': W_ll,
        'LH': W_lh,
        'HL': W_hl,
        'HH': W_hh
    }



def main():
    parser = argparse.ArgumentParser(description='DWT-SVD Watermark Embed/Extract Tool')
    subparsers = parser.add_subparsers(dest='mode', required=True,
                                       help='Mode: embed or extract')

    parser_embed = subparsers.add_parser('embed', help='Embed watermark into cover image')
    parser_embed.add_argument('-c', '--cover', required=True, help='Path to cover image (grayscale)')
    parser_embed.add_argument('-w', '--watermark', required=True, help='Path to watermark image (grayscale, square)')
    parser_embed.add_argument('-o', '--output', required=True, help='Output path for watermarked image')
    parser_embed.add_argument('--alpha_ll', type=float, default=0.05, help='Scaling factor for LL subband')
    parser_embed.add_argument('--alpha_others', type=float, default=0.005, help='Scaling factor for other subbands')
    parser_embed.add_argument('--wavelet', type=str, default='haar', help='Wavelet name for DWT')

    parser_extract = subparsers.add_parser('extract', help='Extract watermark from watermarked image')
    parser_extract.add_argument('-wmi', '--watermarked', required=True, help='Path to watermarked image (grayscale)')
    parser_extract.add_argument('-c', '--cover', required=True, help='Path to original cover image (grayscale)')
    parser_extract.add_argument('-wm', '--watermark', required=True, help='Path to original watermark image (grayscale, square)')
    parser_extract.add_argument('-o', '--output', required=True, help='Output path for extracted watermark')
    parser_extract.add_argument('--alpha_ll', type=float, default=0.05, help='Scaling factor used for LL band')
    parser_extract.add_argument('--alpha_others', type=float, default=0.005, help='Scaling factor used for other bands')
    parser_extract.add_argument('--wavelet', type=str, default='haar', help='Wavelet name used for DWT')

    args = parser.parse_args()

    if args.mode == 'embed':
        cover_img = cv2.imread(args.cover, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        watermark = cv2.imread(args.watermark, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if cover_img is None or watermark is None:
            print('Error: Unable to read cover or watermark image.')
            return
        if cover_img.shape[0] != cover_img.shape[1] or cover_img.shape[0] != 2 * watermark.shape[0]:
            print('Error: Cover image must be square and twice the size of watermark.')
            return
        wm_img = embed_watermark(cover_img, watermark,
                                 alpha_ll=args.alpha_ll, alpha_others=args.alpha_others,
                                 wavelet=args.wavelet)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        cv2.imwrite(args.output, wm_img)
        print(f'Watermarked image saved to {args.output}')

    elif args.mode == 'extract':
        watermarked_img = cv2.imread(args.watermarked, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        cover_img = cv2.imread(args.cover, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        watermark = cv2.imread(args.watermark, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if watermarked_img is None or cover_img is None or watermark is None:
            print('Error: Unable to read one or more input images.')
            return
        
        extracted_dict = extract_watermark(watermarked_img, cover_img, watermark,
                                           alpha_ll=args.alpha_ll, alpha_others=args.alpha_others,
                                           wavelet=args.wavelet)
        prefix = args.output
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        for band, img in extracted_dict.items():
            out_path = f"{prefix}_{band}.png"
            cv2.imwrite(out_path, img)
            print(f'Extracted {band} watermark saved to {out_path}')


if __name__ == '__main__':
    main()
