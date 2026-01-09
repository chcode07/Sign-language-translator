import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# CONFIGURATION 
# Correct path for WSL accessing Windows C: drive (only to be impelmented if dedicated GPU available in the laptop)
DATASET_DIR = "/mnt/c/Users/chann/major_project/augmented_databook"
SEQ_LEN = 90
FEATURE_DIM = 159 

#  1. LOAD DATA 
X, y = [], []
label_map = {}

print(f"\n Loading data from: {DATASET_DIR}...")

if not os.path.exists(DATASET_DIR):
    print(f" ERROR: Directory not found: {DATASET_DIR}")
    exit()

# Iterate through alphabetical folders
for idx, class_name in enumerate(sorted(os.listdir(DATASET_DIR))):
    label_map[idx] = class_name
    class_path = os.path.join(DATASET_DIR, class_name)
    
    if not os.path.isdir(class_path): continue
    
    # Load every .npy file
    files = [f for f in os.listdir(class_path) if f.endswith(".npy")]
    for file in files:
        file_path = os.path.join(class_path, file)
        try:
            seq = np.load(file_path)
            # Integrity check
            if seq.shape == (SEQ_LEN, FEATURE_DIM):
                X.append(seq)
                y.append(idx)
            else:
                print(f" Skipping {file}: Wrong shape {seq.shape}")
        except Exception as e:
            print(f" Error reading {file}: {e}")

X = np.array(X)
y = np.array(y)

print(f"\n Data Loaded Successfully!")
print(f"Total Samples: {len(X)}")
print(f"Labels: {label_map}")

if len(X) == 0:
    print(" ERROR: No data found. Run data_augmentation.py first.")
    exit()

# 2. TRAIN/TEST SPLIT 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#  3. MODEL ARCHITECTURE 
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(SEQ_LEN, FEATURE_DIM)),
    
    # Layer 1
    tf.keras.layers.GRU(64, return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    
    # Layer 2
    tf.keras.layers.GRU(32, return_sequences=False),
    tf.keras.layers.Dropout(0.3),
    
    # Layer 3
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(label_map), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# 4. CALLBACKS 
callbacks = [
    ModelCheckpoint(
        "signlanguage_model.keras", 
        save_best_only=True, 
        monitor="val_loss", 
        mode="min"
    ),
    
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=1)
]

# 5. START TRAINING 
print("\n Starting Training...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# 6. SAVE ARTIFACTS 
# Save label map
np.save("label_map.npy", label_map)

# Explicitly save final model as .keras
model.save("signlanguage_model.keras")

print("\n🎉 Training Complete!")
print("Model saved: signlanguage_model.keras")
print("Labels saved: label_map.npy")