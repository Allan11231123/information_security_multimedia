import os
from PIL import Image
import numpy as np
from argparse import ArgumentParser
from .utils import base64_to_image, image_to_base64_string
from .embed_utils import embed_message, extract_message

def main():
    parser = ArgumentParser(description="FFT Steganography Example")
    parser.add_argument('--cover', type=str, default='cover.png', help='Path to cover image')
    parser.add_argument('--stego', type=str, default='stego.png', help='Name for output stego image')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save output files')
    parser.add_argument('--method', type=str, default='magnitude', choices=['magnitude', 'phase', 'hamming'], help='Methods to use for embedding (e.g., fft)')
    parser.add_argument('--input_type', type=str, default='text', choices=['image', 'text'], help='Type of input data (e.g., image, text)')
    parser.add_argument('--input_file', type=str, default='input.txt', help='Path to input file for embedding')
    args = parser.parse_args()
    key = os.urandom(16)  # Generate a random 128-bit AES key
    cover_image = args.cover
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    stego_image = os.path.join(args.output_dir, args.stego)
    # check embedding method
    if args.method not in ['magnitude', 'phase', 'hamming']:
        raise ValueError("Invalid embedding method. Choose from: magnitude, phase, hamming.")
    if args.input_type not in ['image', 'text']:
        raise ValueError("Invalid input type. Choose from: image, text.")
    if args.method ==  'magnitude':
        from .embed_utils import embed_message as embed_func
        from .embed_utils import extract_message as extract_func
    elif args.method == 'phase':
        from .embed_utils import embed_phase_coeff as embed_func
        from .embed_utils import extract_phase_coeff as extract_func
    elif args.method == 'hamming':
        from .embed_utils import embed_hamming as embed_func
        from .embed_utils import extract_hamming as extract_func
    # check input_type and read secret message
    if args.input_type == 'text':
        with open(args.input_file, 'r') as f:
            secret_message = f.read().strip()
    elif args.input_type == 'image':
        secret_message = image_to_base64_string(args.input_file)

    # Load cover image as grayscale array
    img = Image.open(cover_image).convert('L')
    arr = np.array(img, dtype=np.float32)
    # Embed the secret message into the cover image
    bit_length = embed_func(cover_image=arr, message=secret_message, key=key, output_path=stego_image)
    message = extract_func(stego_path=stego_image, key=key, bit_len=bit_length)
    if args.input_type == 'image':
        base64_to_image(message, os.path.join(args.output_dir, 'recovered_image.png'))
        print("Recovered image saved as 'recovered_image.png'")
    else:
        print("Recovered message:\n", message)
    return
# Example usage
if __name__ == '__main__':
    main()

