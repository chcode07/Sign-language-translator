import numpy as np
import os

# ------------------------------------------------------------
# FEATURE-SPACE AUGMENTATION
# sequence shape: (frames, features) = (T, 159)
# ------------------------------------------------------------

def augment_landmarks(sequence, num_variations=10):
    """
    Generate augmented variations of a landmark sequence.

    Parameters
    ----------
    sequence : np.ndarray
        Shape (frames, features) -> (T, 159)
    num_variations : int
        Number of augmented samples to generate

    Returns
    -------
    List[np.ndarray]
        Each array has same shape as input sequence
    """

    augmented_data = []
    frames, features = sequence.shape

    for _ in range(num_variations):
        aug = sequence.copy()

        # 1. Gaussian noise (sensor jitter)
        noise = np.random.normal(0, 0.002, aug.shape)
        aug += noise

        # 2. Global scaling (distance variation)
        scale = np.random.uniform(0.9, 1.1)
        aug *= scale

        # 3. Temporal shift (timing variation)
        if frames > 1:
            shift = np.random.randint(-2, 3)
            aug = np.roll(aug, shift, axis=0)

        augmented_data.append(aug.astype(np.float32))

    return augmented_data


# ------------------------------------------------------------
# DATASET EXPANSION LOOP
# ------------------------------------------------------------

DATA_ROOT = "C:/Users/chann/major_project/data_book"

for folder in os.listdir(DATA_ROOT):

    folder_path = os.path.join(DATA_ROOT, folder)
    original_path = os.path.join(folder_path, "video_0.npy")

    if not os.path.isfile(original_path):
        print(f"Skipping {folder} (no video_0.npy)")
        continue

    # Load original landmark sequence
    original_clip = np.load(original_path)

    print(f"{folder} | original shape:", original_clip.shape)

    # Generate augmented samples
    new_samples = augment_landmarks(original_clip, num_variations=50)

    # Save augmented samples (video_1.npy ... video_50.npy)
    for idx, sample in enumerate(new_samples, start=1):
        out_path = os.path.join(folder_path, f"video_{idx}.npy")
        np.save(out_path, sample)

    print(f"{folder}: generated {len(new_samples)} augmented samples\n")
