from PIL import Image
from tqdm import tqdm
import numpy as np
from utils import encrypt_message, decrypt_message, get_mid_freq_coords

def embed_message(
    cover_image: np.ndarray,
    message: str,
    key: bytes,
    output_path: str = 'stego.png',
    **kwargs
):
    if 'delta' not in kwargs:
        delta = 2.0  # Default delta value for magnitude embedding
    else:
        delta = kwargs['delta']
    if 'low_frac' in kwargs:
        low_frac = kwargs['low_frac']
    else:
        low_frac = 0.1
    if 'high_frac' in kwargs:
        high_frac = kwargs['high_frac']
    else:
        high_frac = 0.5
    # Load cover image as grayscale array
    # img = Image.open(cover_path).convert('L')
    # arr = np.array(img, dtype=np.float32)
    arr = cover_image.copy()
    # Compute 2D FFT
    F = np.fft.fft2(arr)
    F_shifted = np.fft.fftshift(F) # shift zero frequency to center
    A = np.abs(F_shifted) # Magnitude spectrum
    P = np.angle(F_shifted) # Phase spectrum

    # Encrypt message and get bit array
    ct = encrypt_message(message, key)
    bits = np.unpackbits(np.frombuffer(ct, dtype=np.uint8))
    # print(f"length for bit array: {len(bits)}, bits: \n{bits}")
    # Get bit length for returning
    bit_length = len(bits)

    # Select mid-frequency coords
    coords = get_mid_freq_coords(arr.shape,low_frac=low_frac, high_frac=high_frac)
    print(f"coords:\n{coords[:bit_length]}")
    # print(f"low_frac: {low_frac}, high_frac: {high_frac}")
    if len(bits) > len(coords):
        raise ValueError(f"Message too long for available frequency coefficients, difference {len(bits) - len(coords)}, try increase the `high_frac` parameter or reduce the message size")

    # QIM embedding
    for idx, bit in tqdm(enumerate(bits), desc="Embedding bits"):
        i, j = coords[idx]
        a = A[i, j]
        q = np.round((a + delta/2) / delta) * delta
        if bit == 1:
            a_prime = q + delta/4
        else:
            a_prime = q - delta/4
        A[i, j] = a_prime
    # amplitude_new = [A[i, j] for i, j in coords[:bit_length]]
    # print(f"amplitude_new:\n{amplitude_new}")

    # Reconstruct and save stego image
    F_prime = A * np.exp(1j * P)
    stego_arr = np.real(np.fft.ifft2(np.fft.ifftshift(F_prime)))
    stego_clipped = np.clip(stego_arr, 0, 255).astype(np.uint8)
    Image.fromarray(stego_clipped).save(output_path)
    print(f"[QIM embedding(magnitude)] - Stego image saved to: {output_path}, length for bits array is {bit_length}")
    return bit_length

def extract_message(
    stego_path: str,
    key: bytes,
    bit_len: int,
    **kwargs,
) -> str:
    if 'delta' not in kwargs:
        delta = 2.0
    else:
        delta = kwargs['delta']
    if 'low_frac' in kwargs:
        low_frac = kwargs['low_frac']
    else:
        low_frac = 0.1
    if 'high_frac' in kwargs:
        high_frac = kwargs['high_frac']
    else:
        high_frac = 0.5
    # print(f"Received bit length: {bit_len}, delta: {delta}")
    # Load stego image
    img = Image.open(stego_path).convert('L')
    arr = np.array(img, dtype=np.float32)

    # Compute FFT
    F = np.fft.fft2(arr)
    F_shifted = np.fft.fftshift(F)  # shift zero frequency to center
    A = np.abs(F_shifted)
    # Select same mid-frequency coords
    coords = get_mid_freq_coords(arr.shape, low_frac=low_frac, high_frac=high_frac)
    # amplitude_recv = [A[i, j] for i, j in coords[:bit_len]]
    # print(f"coords:\n{coords[:bit_len]}")
    # print(f"amplitude_recv:\n{amplitude_recv}")
    original_high = high_frac
    # print(f"Number of mid-frequency coefficients available: {len(coords)}")
    while len(coords) < bit_len:
        original_high += 0.1
        coords = get_mid_freq_coords(arr.shape, low_frac=low_frac, high_frac=original_high)
        if original_high > 0.9:
            print("Not enough mid-frequency coefficients available")
            break
    # print(f"low_frac: {low_frac}, high_frac: {original_high}")
    # if bit_len > len(coords):
    #     raise ValueError("Requested more bits than available coefficients")

    # Extract bits via QIM decoding
    bits = []
    for idx in range(bit_len):
        i, j = coords[idx]
        a = A[i, j]
        q = np.round((a + delta/2) / delta) * delta
        bits.append(1 if (q - a) >= 1 else 0)
    print(f"length for recovered bits: {len(bits)}, bits: \n{bits}")
    # Reconstruct ciphertext and decrypt
    byte_arr = np.packbits(bits[:bit_len])
    ct = byte_arr.tobytes()
    # return ct.decode('utf-8')
    message = decrypt_message(ct, key)
    return message

# ------------------------------------------------------------------
# Phase‑coding embed / extract
# ------------------------------------------------------------------
def embed_phase_coeff(
    cover_image: np.ndarray,
    message: str,
    key: bytes,
    output_path: str = 'stego.png',
    **kwargs
) -> int:
    """
    Embed 1 bit / coefficient by snapping its phase to one of two quantisation bins
    of width ``step`` radians (default π/8).

    Only the phase is modified → amplitude is preserved → very high visual fidelity.
    """
    if 'step' not in kwargs:
        step = np.pi / 8
    else:
        step = kwargs['step']
    if 'low_frac' in kwargs:
        low_frac = kwargs['low_frac']
    else:
        low_frac = 0.1
    if 'high_frac' in kwargs:
        high_frac = kwargs['high_frac']
    else:
        high_frac = 0.5
    arr = cover_image.copy()
    F = np.fft.fft2(arr)
    F_mod = np.fft.fftshift(F)  # shift zero frequency to center
    # print(f"F_mod: \n{F_mod}")
    # A = np.abs(F_mod)  # Magnitude spectrum
    # P = np.angle(F_mod)  # Phase spectrum
    # print(f"A: \n{A}")
    # print(f"P: \n{P}")
    bits = np.unpackbits(np.frombuffer(encrypt_message(message, key), dtype=np.uint8))
    coords = get_mid_freq_coords(arr.shape, low_frac=low_frac, high_frac=high_frac)
    # print(f"coords:\n{coords[:len(bits)]}")
    print(f"length for bit array: {len(bits)}, bits: \n{bits}")
    if len(bits) > len(coords):
        raise ValueError("Message too long for available frequency coefficients, \
            try increase the `high_frac` parameter or reduce the message size")
    for b, (i, j) in zip(bits, coords):
        amp = np.abs(F_mod[i, j])
        phase = np.angle(F_mod[i, j])
        # wrap to [0, 2π)
        phase = (phase + 2 * np.pi) % (2 * np.pi)
        # print(phase)
        # quantisation
        base = np.floor(phase / step) * step
        new_phase = base + (step / 2 if b else 0.0)
        # keep within [0, 2π)
        new_phase = new_phase % (2 * np.pi)
        F_mod[i, j] = amp * np.exp(1j * new_phase)
    # phase_new = [np.angle(F_mod[i, j]) for i, j in coords[:len(bits)]]
    # print(f"phase_new:\n{phase_new}")
    
    stego_arr = np.real(np.fft.ifft2(np.fft.ifftshift(F_mod)))
    stego_clipped = np.clip(stego_arr, 0, 255).astype(np.uint8)
    Image.fromarray(stego_clipped).save(output_path)
    print(f"[QIM embedding(phase)] - Stego image saved to: {output_path}")
    return len(bits)


def extract_phase_coeff(
    stego_path: str,
    key: bytes,
    bit_len: int,
    **kwargs,
) -> str:
    """
    Reverse of ``embed_phase_coeff``.
    """
    if 'step' not in kwargs:
        step = np.pi / 8
    else:
        step = kwargs['step']
    img = Image.open(stego_path).convert('L')
    arr = np.array(img, dtype=np.float32)
    F = np.fft.fft2(arr)
    F_new = np.fft.fftshift(F)  # shift zero frequency to center
    coords = get_mid_freq_coords(arr.shape)
    print(f"coords:\n{coords[:bit_len]}")
    original_high = 0.5
    while len(coords) < bit_len:
        original_high += 0.1
        coords = get_mid_freq_coords(arr.shape, low_frac=0.1, high_frac=original_high)
        if original_high > 0.9:
            print("Not enough mid-frequency coefficients available")
            break
    # if bit_len > len(coords):
    #     raise ValueError("Requested more bits than available coefficients")
    bits = []
    for (i, j) in coords[:bit_len]:
        phase = (np.angle(F_new[i, j]) + 2 * np.pi) % (2 * np.pi)
        bits.append(1 if (phase % step) >= (step / 2) else 0)
    print(f"length for recovered bits: {len(bits)}, bits: \n{bits}")
    # phase_recv = [np.angle(F_new[i, j]) for i, j in coords[:bit_len]]
    # print(f"phase_recv:\n{phase_recv}")
    # Convert bits to bytes
    byte_arr = np.packbits(bits[:bit_len])
    ct = byte_arr.tobytes()
    message = decrypt_message(ct, key)

    return message


# ------------------------------------------------------------------
# Parity / matrix‑encoding (Hamming‑(7,4)) embed / extract
# ------------------------------------------------------------------
def _hamming_syndrome(bits7: np.ndarray) -> np.ndarray:
    """Return the 3‑bit syndrome for (7,4) Hamming parity matrix."""
    H = np.array([[1, 0, 1, 0, 1, 0, 1],
                  [0, 1, 1, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1]], dtype=int)
    return (H @ bits7) % 2


def embed_hamming(
    cover_image: np.ndarray,
    message: str,
    key: bytes,
    output_path: str = 'stego.png',
    **kwargs,
) -> int:
    """
    Embed **4 payload bits into every 7 coefficients** using Hamming‑(7,4).
    Matrix encoding flips at most *one* coefficient LSB in each 7‑tuple,
    ⇒ average distortion ↓ 3.5× vs naive LSB.

    ``delta`` controls the minimal amplitude nudge used to flip an LSB.
    """
    if 'delta' not in kwargs:
        delta = 1.0
    else:
        delta = kwargs['delta']
    if 'low_frac' in kwargs:
        low_frac = kwargs['low_frac']
    else:
        low_frac = 0.1
    if 'high_frac' in kwargs:
        high_frac = kwargs['high_frac']
    else:
        high_frac = 0.5
    arr = cover_image.copy()
    F = np.fft.fft2(arr)
    F_mod = np.fft.fftshift(F)  # shift zero frequency to center
    bits = np.unpackbits(np.frombuffer(encrypt_message(message,key), dtype=np.uint8))
    coords = get_mid_freq_coords(arr.shape, low_frac=low_frac, high_frac=high_frac)
    
    bit_idx = 0
    for blk_start in range(0, len(coords), 7):
        blk = coords[blk_start: blk_start + 7]
        if bit_idx >= len(bits):
            break
        payload = bits[bit_idx: bit_idx + 4]
        bit_idx += 4

        # current LSBs of amplitude (could use magnitude mod 2*delta instead)
        amp = np.abs([F_mod[i, j] for i, j in blk])
        lsb = np.round(amp / delta) % 2

        # desired codeword: append parity to payload
        codeword = np.zeros(7, dtype=int)
        codeword[:4] = payload
        # compute parity bits to satisfy Hamming equation
        # simple generator for (7,4): parity bits at positions 3,5,6 (0‑index)
        codeword[4] = (codeword[0] ^ codeword[1] ^ codeword[2])
        codeword[5] = (codeword[0] ^ codeword[1] ^ codeword[3])
        codeword[6] = (codeword[0] ^ codeword[2] ^ codeword[3])

        # difference between current and desired ⇒ syndrome
        syndrome = _hamming_syndrome(lsb.astype(int) ^ codeword)
        err_pos = int("".join(map(str, syndrome)), 2) - 1  # 0‑based
        if err_pos >= 0:
            # flip one coefficient by ±delta to match
            i, j = blk[err_pos]
            F_mod[i, j] += delta if (np.random.rand() > 0.5) else -delta
    # Reconstruct and save stego image
    stego_arr = np.real(np.fft.ifft2(np.fft.ifftshift(F_mod)))
    stego_clipped = np.clip(stego_arr, 0, 255).astype(np.uint8)
    Image.fromarray(stego_clipped).save(output_path)
    print(f"[Hamming embedding] - Stego image saved to: {output_path}")
    return len(bits)


def extract_hamming(
    stego_path: str,
    key: bytes,
    bit_len: int,
    **kwargs,
) -> str:
    """
    Extract payload bits from Hamming‑(7,4) encoded coefficients.
    """
    if 'delta' not in kwargs:
        delta = 1.0
    else:
        delta = kwargs['delta']
    img = Image.open(stego_path).convert('L')
    arr = np.array(img, dtype=np.float32)
    F = np.fft.fft2(arr)
    F = np.fft.fftshift(F)  # shift zero frequency to center
    coords = get_mid_freq_coords(arr.shape)
    original_high = 0.5
    while len(coords) < bit_len:
        original_high += 0.1
        coords = get_mid_freq_coords(arr.shape, low_frac=0.1, high_frac=original_high)
        if original_high > 0.9:
            print("Not enough mid-frequency coefficients available")
            break
    bits_out = []
    for blk_start in range(0, min(len(coords), (bit_len // 4 + 1) * 7), 7):
        blk = coords[blk_start: blk_start + 7]
        amp = np.abs([F[i, j] for i, j in blk])
        lsb = (np.round(amp / delta) % 2).astype(int)
        # The first 4 bits of the codeword are the payload
        bits_out.extend(lsb[:4])
        if len(bits_out) >= bit_len:
            break
    # Convert bits to bytes
    bits_out = np.array(bits_out, dtype=np.uint8)
    byte_arr = np.packbits(bits_out[:bit_len])
    ct = byte_arr.tobytes()
    message = decrypt_message(ct, key)
    return message


# ------------------------------------------------------------------
# Spread‑spectrum embed / extract (binary PN sequence)
# ------------------------------------------------------------------
def embed_spread_spectrum(
    cover_image: np.ndarray,
    message: str,
    key: bytes,
    output_path: str = 'stego.png',
    **kwargs,
) -> int:
    """
    Adds a tiny pseudo‑random ±alpha * mean(|F|) amplitude to *all* chosen coefficients.
    Each payload bit is spread across the full coords set ⇒ high robustness.

    Only one bit is embedded per call; repeat for the whole message.
    """
    if 'alpha' not in kwargs:
        alpha = 0.05
    else:
        alpha = kwargs['alpha']
    arr = cover_image.copy()
    F = np.fft.fft2(arr)
    F_mod = np.fft.fftshift(F)  # shift zero frequency to center
    bits = np.unpackbits(np.frombuffer(encrypt_message(message, key), dtype=np.uint8))
    coords = get_mid_freq_coords(arr.shape)
    if len(bits) > len(coords):
        raise ValueError("Message too long for available frequency coefficients")
    # Convert bytes to integer for seed
    seed_int = int.from_bytes(key, byteorder='little')
    rng = np.random.default_rng(seed_int)
    pn = rng.choice([-1, 1], size=len(coords))
    mean_amp = np.mean(np.abs([F[i, j] for i, j in coords]))

    for b in bits:
        shift = alpha * mean_amp * (1 if b else -1)
        for n, (i, j) in enumerate(coords):
            F_mod[i, j] += shift * pn[n]
    # Reconstruct and save stego image
    stego_arr = np.real(np.fft.ifft2(np.fft.ifftshift(F_mod)))
    stego_clipped = np.clip(stego_arr, 0, 255).astype(np.uint8)
    Image.fromarray(stego_clipped).save(output_path)
    print(f"[Spread-spectrum embedding] - Stego image saved to: {output_path}")
    return len(bits)


def extract_spread_spectrum(
    stego_path: str,
    key: bytes,
    bit_len: int,
    **kwargs,
) -> str:
    """
    Demodulate Spread‑spectrum bit stream via correlation with the PN sequence.
    """
    if 'alpha' not in kwargs:
        alpha = 0.05
    img = Image.open(stego_path).convert('L')
    arr = np.array(img, dtype=np.float32)
    F = np.fft.fft2(arr)
    F = np.fft.fftshift(F)  # shift zero frequency to center
    coords = get_mid_freq_coords(arr.shape)
    # Convert bytes to integer for seed
    seed_int = int.from_bytes(key, byteorder='little')
    rng = np.random.default_rng(seed_int)
    pn = rng.choice([-1, 1], size=len(coords))
    bits_out = []
    # Extract bits by correlating with the PN sequence
    for _ in range(bit_len):
        corr = np.sum([
            pn[n] * np.abs(F[i, j]) for n, (i, j) in enumerate(coords)
        ])
        bits_out.append(1 if corr >= 0 else 0)
    # Convert bits to bytes
    bits_out = np.array(bits_out, dtype=np.uint8)
    byte_arr = np.packbits(bits_out[:bit_len])
    ct = byte_arr.tobytes()
    message = decrypt_message(ct, key)
    return message
