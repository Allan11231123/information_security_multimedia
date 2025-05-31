from PIL import Image
import numpy as np
from .utils import encrypt_message, decrypt_message
def _get_mid_freq_coords(
        shape, 
        low_frac=0.1, 
        high_frac=0.5
):
    """
    Returns a list of mid-frequency (i, j) coordinates ordered by distance from the center,
    skipping DC at (0,0), and selecting indices between low_frac and high_frac of the sorted list.
    """
    h, w = shape
    center = np.array([h/2, w/2])
    coords = [(i, j) for i in range(h) for j in range(w) if not (i == 0 and j == 0)]
    coords_sorted = sorted(coords, key=lambda x: np.hypot(x[0]-center[0], x[1]-center[1]))
    start = int(len(coords_sorted) * low_frac)
    end = int(len(coords_sorted) * high_frac)
    return coords_sorted[start:end]

def embed_message(
    cover_path: str,
    message: str,
    key: bytes,
    delta: float = 2.0,
    output_path: str = 'stego.png'
):
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

    if len(bits) > len(coords):
        raise ValueError("Message too long for available frequency coefficients")

    # QIM embedding
    for idx, bit in enumerate(bits):
        i, j = coords[idx]
        a = A[i, j]
        q = np.round((a + delta/2) / delta) * delta
        if bit == 1:
            a_prime = q + delta/4
        else:
            a_prime = q - delta/4
        A[i, j] = a_prime

    # Reconstruct and save stego image
    F_prime = A * np.exp(1j * P)
    stego_arr = np.real(np.fft.ifft2(F_prime))
    stego_clipped = np.clip(stego_arr, 0, 255).astype(np.uint8)
    Image.fromarray(stego_clipped).save(output_path)
    print(f"Stego image saved to: {output_path}")
    return bit_length

def extract_message(
    stego_path: str,
    key: bytes,
    bit_len: int,
    delta: float = 2.0
) -> str:
    # Load stego image
    img = Image.open(stego_path).convert('L')
    arr = np.array(img, dtype=np.float32)

    # Compute FFT
    F = np.fft.fft2(arr)
    F_shifted = np.fft.fftshift(F)  # shift zero frequency to center
    A = np.abs(F_shifted)

    # Select same mid-frequency coords
    coords = _get_mid_freq_coords(arr.shape)

    if bit_len > len(coords):
        raise ValueError("Requested more bits than available coefficients")

    # Extract bits via QIM decoding
    bits = []
    for idx in range(bit_len):
        i, j = coords[idx]
        a = A[i, j]
        q = np.round((a + delta/2) / delta) * delta
        bits.append(1 if (a - q) >= 0 else 0)

    # Reconstruct ciphertext and decrypt
    byte_arr = np.packbits(bits[:bit_len])
    ct = byte_arr.tobytes()
    message = decrypt_message(ct, key)
    return message
