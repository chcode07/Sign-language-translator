import numpy as np
import os
from tqdm import tqdm

# --- CONFIG ---
INPUT_DIR = "./data_book_1"               
AUGMENTED_DIR = "./augmented_databook"  
NUM_VARIATIONS = 50                   
TARGET_LEN = 90  # Force everything to this length

def resize_sequence(sequence, target_len=90):
 
    orig_len, num_features = sequence.shape
    
    if orig_len == target_len:
        return sequence
    
    # Create indices
    old_indices = np.arange(orig_len)
    new_indices = np.linspace(0, orig_len - 1, target_len)
    
    # Interpolate each feature
    resized_seq = np.zeros((target_len, num_features))
    for i in range(num_features):
        resized_seq[:, i] = np.interp(new_indices, old_indices, sequence[:, i])
        
    return resized_seq

def augment_sequence(sequence):
  
    # 1. Jitter
    aug_seq = sequence.copy()
    noise = np.random.normal(0, 0.02, aug_seq.shape)
    aug_seq += noise

    # 2. Scaling
    scale_factor = np.random.uniform(0.9, 1.1)
    aug_seq *= scale_factor

    # 3. Time Warp (Random speed change)
    orig_len = aug_seq.shape[0]
    speed_factor = np.random.uniform(0.8, 1.2)
    new_temp_len = int(orig_len * speed_factor)
    
    # Resize to the warped length first (simulates speed change)
    aug_seq = resize_sequence(aug_seq, new_temp_len)
    
    # 4. FINAL FORCE RESIZE to 90
    # No matter what happened above, we return 90 frames.
    final_seq = resize_sequence(aug_seq, TARGET_LEN)
        
    return final_seq

#  EXECUTION LOOP 
if not os.path.exists(AUGMENTED_DIR):
    os.makedirs(AUGMENTED_DIR)

print(f"Starting Augmentation: {INPUT_DIR} -> {AUGMENTED_DIR}")

for class_name in os.listdir(INPUT_DIR):
    class_path = os.path.join(INPUT_DIR, class_name)
    if not os.path.isdir(class_path): continue
    
    aug_class_path = os.path.join(AUGMENTED_DIR, class_name)
    os.makedirs(aug_class_path, exist_ok=True)
    
    print(f"  Processing Class: {class_name}...")
    
    for file_name in os.listdir(class_path):
        if file_name.endswith(".npy"):
            # Load original (Variable Shape, e.g., 143, 159)
            original_seq = np.load(os.path.join(class_path, file_name))
            
            # 1. Save ORIGINAL (Resized to 90)
            orig_fixed = resize_sequence(original_seq, TARGET_LEN)
            np.save(os.path.join(aug_class_path, f"orig_{file_name}"), orig_fixed)
            
            # 2. Save VARIATIONS (Resized to 90)
            for i in range(NUM_VARIATIONS):
                new_seq = augment_sequence(original_seq)
                np.save(os.path.join(aug_class_path, f"aug_{i}_{file_name}"), new_seq)

print("\n Augmentation Complete!")
print("All files are now guaranteed to be shape (90, 159).")
print("Run train.py now.")