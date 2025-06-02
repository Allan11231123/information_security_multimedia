#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_metrics(orig_path: str, stego_path: str, use_grayscale: bool = False):
    orig = cv2.imread(orig_path, cv2.IMREAD_COLOR)
    stego = cv2.imread(stego_path, cv2.IMREAD_COLOR)

    if orig is None:
        raise FileNotFoundError(f"can't find original image: {orig_path}")
    if stego is None:
        raise FileNotFoundError(f"can't find Stego image: {stego_path}")

    if orig.shape != stego.shape:
        raise ValueError(f"Dimension mismatch: {orig.shape} vs {stego.shape}")

    if use_grayscale:
        orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        stego_gray = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)

        psnr_val = peak_signal_noise_ratio(orig_gray, stego_gray, data_range=255)

        ssim_val = structural_similarity(orig_gray, stego_gray, data_range=255)

        return psnr_val, ssim_val

    else:
        psnr_val = peak_signal_noise_ratio(orig, stego, data_range=255)

        ssim_val, ssim_map = structural_similarity(
            orig,
            stego,
            data_range=255,
            multichannel=True,
            full=True
        )
        return psnr_val, ssim_val

def main():
    parser = argparse.ArgumentParser(description="Script for evaluating image quality metrics (PSNR, SSIM) between original and stego images.")
    parser.add_argument("orig",  help="Path to the original image file")
    parser.add_argument("stego", help="Path to the stego image file")
    parser.add_argument(
        "--gray",
        action="store_true",
        help="If specified, convert images to grayscale before computing metrics (PSNR, SSIM). Otherwise, use color images."
    )
    args = parser.parse_args()

    psnr_val, ssim_val = compute_metrics(args.orig, args.stego, use_grayscale=args.gray)

    print(f"=== Result for evaluation ===")
    print(f"Original image: {args.orig}")
    print(f"Stego image: {args.stego}")
    if args.gray:
        print("Using grayscale for PSNR / SSIM")
    else:
        print("Using color images for PSNR / SSIM")
    print(f"PSNR: {psnr_val:.4f} dB")
    print(f"SSIM: {ssim_val:.4f}")

if __name__ == "__main__":
    main()
