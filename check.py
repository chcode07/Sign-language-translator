#!/usr/bin/env python3

# python inspect_sequence.py <numpy_file_path>


import sys
import numpy as np

def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_sequence.py <path_to_npy_file>")
        sys.exit(1)

    npy_path = sys.argv[1]

    sequence = np.load(npy_path)

    print("Sequence shape:", sequence.shape)
    print("\nFirst 10 values of first frame:")
    print(sequence[0][:10])

    print("\nMean std across features:")
    print(sequence.std(axis=0).mean())


if __name__ == "__main__":
    main()
