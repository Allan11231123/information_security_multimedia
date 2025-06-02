# information_security_multimedia

## FFT + QIM (on magnitude)
- steps:
    - Encrypt the input text into ciphertext.(AES)
    - Conduct 2D FFT on the target image and divide the magnitude and phase.
    - Pick the medium-range frequency for embedding(under QIM).
    - Recover the image (IFFT).
    - Using the same key and argument to extract the original text.
- [This survey](https://magrf.grf.hr/wp-content/uploads/2022/07/LOW-FREQUENCY-DATA-EMBEDDING-FOR-DFT-BASED-IMAGE-STEGANOGRAPHY-1.pdf) shows that the medium range of frequency has better robustness on detection rate on embeddings.

## FFT + Hamming (7,4) Matrix embedding
- Intro:
    - Hamming(7,4) is a classical linear grouping correcting code, it maps 4-bit data $d = (d_1,d_2,d_3,d_4)$ to a 7-bit code $c = (c_1,c_2,...,c_7)$. The 4 bits in the front is the `information bits`, while the last 3 bits stand for `parity bits`, which used to detect error and correct for at most 1 bit mismatch.
    - After embedding, the receiver can use parity-check matrix to fast locate at the location where the bit is reversed and correct it to produce the original code.

## FFT + QIM (on phase) (Deprecated: texture will be modified after the embedding)
- steps similar to the original one
- Difference: Target on the phase field to reduce the impact to the imaging for the input cover image.
    - Effect: Since human eyes more focus on the amplitude of the frequency, changing the phase field can reduce the difference between cover and stego image.


## SSIS (Spread-Spectrum) (Deprecated: need mutiple PN sequence to embed multiple bits)
- steps:
    - Generate a pseudo-noise sequence(PN sequence)
    - Based on the PN sequence, which has the same length as the selected frequency sequence, and the bit array, calculate the new amplitude for the FFT map.
    $$PN = \{p_1, p_2, ..., p_n\}, \quad p_i \in \{+1,-1\}$$
    $$A^\prime(u_i, v_i) = A(u_i, v_i) + \alpha \cdot p_i \cdot (2b_k - 1) \cdot \mu, \quad b_k \in \{ 0, 1\}$$
        - $\alpha \approx$ 0.01~0.05, $\alpha$ is an amplify coefficient.
        - $\mu$ is the average of the original amplitude sequence, $\mu = \frac{1}{N}\sum^N_{i=1}A(u_i, u_v)$
        - $(2b_k-1) \in \{+1,-1\}$, where $b_k \in \{1,0\}$
    - When extracting the cipher text, using the same PN sequence $PN = \{p_1, p_2, ..., p_n\}, \quad p_i \in \{+1,-1\}$
        - first calculate the amplitude sequence $A_{recv}(u_i,v_i)$
        - then calculate the `related coefficient`,
        $$C = \sum_{i=1}^Np_i \cdot A_{recv}(u_i,v_i)$$
        if the $C \geq 0$, then $b_k=1$ else $b_k=0$
