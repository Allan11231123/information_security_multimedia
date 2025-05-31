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
