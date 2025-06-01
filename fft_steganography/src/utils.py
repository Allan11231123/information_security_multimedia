from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

def encrypt_message(message: str, key: bytes) -> bytes:
    """
    Encrypts the input UTF-8 string using AES-CBC and returns IV || ciphertext.
    """
    cipher = AES.new(key, AES.MODE_CBC)
    ct = cipher.encrypt(pad(message.encode('utf-8'), AES.block_size))
    return bytes(cipher.iv) + ct

def decrypt_message(ct_all: bytes, key: bytes) -> str:
    """
    Decrypts input IV || ciphertext using AES-CBC and returns the original UTF-8 string.
    """
    iv = ct_all[:AES.block_size]
    ct = ct_all[AES.block_size:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), AES.block_size).decode('utf-8')

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
