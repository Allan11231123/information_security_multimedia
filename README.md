# information_security_multimedia

## FFT + QIM (on magnitude)
- steps:
    - Encrypt the input text into ciphertext.(AES)
    - Conduct 2D FFT on the target image and divide the magnitude and phase.
    - Pick the medium-range frequency for embedding(under QIM).
    - Recover the image (IFFT).
    - Using the same key and argument to extract the original text.
- [This survey](https://magrf.grf.hr/wp-content/uploads/2022/07/LOW-FREQUENCY-DATA-EMBEDDING-FOR-DFT-BASED-IMAGE-STEGANOGRAPHY-1.pdf) shows that the medium range of frequency has better robustness on detection rate on embeddings.
