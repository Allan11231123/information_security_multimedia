import os
from PIL import Image
import numpy as np
from .utils import encrypt_message, decrypt_message, _get_mid_freq_coords
from .embed_utils import embed_message, extract_message





# Example usage
if __name__ == '__main__':
    key = os.urandom(16)  # 128-bit AES key
    cover = 'cover.png'
    stego = 'stego.png'
    secret = 'Hello, FFT Stego!'

    # Load cover image as grayscale array
    img = Image.open(cover_path).convert('L')
    arr = np.array(img, dtype=np.float32)

    # Compute 2D FFT
    F = np.fft.fft2(arr)
    F_shifted = np.fft.fftshift(F) # shift zero frequency to center
    A = np.abs(F_shifted) # Magnitude spectrum
    P = np.angle(F_shifted) # Phase spectrum

    # Encrypt message and get bit array
    ct = encrypt_message(message, key)
    bits = np.unpackbits(np.frombuffer(ct, dtype=np.uint8))
    # Get bit length for returning
    bit_length = len(bits)

    # Select mid-frequency coords
    coords = _get_mid_freq_coords(arr.shape)

    # Embed
    bit_length = embed_message(cover_path=cover, message=secret, key=key, delta=2.0, output_path=stego)

    # Extract
    recovered = extract_message(stego_path=stego, key=key, bit_len=bit_length, delta=2.0)
    print("Recovered message:", recovered)
