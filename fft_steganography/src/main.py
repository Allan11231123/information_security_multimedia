import os
from .embed_utils import embed_message, extract_message





# Example usage
if __name__ == '__main__':
    key = os.urandom(16)  # 128-bit AES key
    cover = 'cover.png'
    stego = 'stego.png'
    secret = 'Hello, FFT Stego!'

    # Embed
    bit_length = embed_message(cover_path=cover, message=secret, key=key, delta=2.0, output_path=stego)

    # Extract
    recovered = extract_message(stego_path=stego, key=key, bit_len=bit_length, delta=2.0)
    print("Recovered message:", recovered)
