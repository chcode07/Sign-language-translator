import tensorflow as tf
import numpy as np
import os

# --- CONFIGURATION ---
KERAS_MODEL_PATH = "C:/Users/chann/major_project/GRU_model/signlanguage_model.keras"  # Or .keras
TFLITE_MODEL_PATH = "C:/Users/chann/major_project/GRU_model/gru_model.tflite"

def convert_model():
    print(f"Loading model from {KERAS_MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    except OSError:
        print("Error: Model file not found. Check the path.")
        return

    # 1. Initialize Converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 2. ENABLE SELECT TF OPS (Crucial for GRU/LSTM)
    # This allows TFLite to use standard TensorFlow operations if a 
    # specific mobile version doesn't exist. Without this, GRU often fails.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Use mobile-optimized ops where possible
        tf.lite.OpsSet.SELECT_TF_OPS    # Fallback to standard TF ops for GRU
    ]

    # 3. OPTIONAL: Optimization (Makes model 4x smaller)
    # This converts weights from Float32 to Int8, reducing size with minimal accuracy loss.
    # If your accuracy drops too much, comment this line out.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("Converting model... (This might take a moment)")
    tflite_model = converter.convert()

    # 4. Save the file
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)
    
    print(f"✅ Success! Saved to {TFLITE_MODEL_PATH}")
    print(f"Model Size: {len(tflite_model) / 1024:.2f} KB")

    # --- VERIFICATION STEP ---
    # We run a dummy prediction to make sure the .tflite file actually works
    print("\nVerifying model integrity...")
    try:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Create dummy input (Batch Size 1, 30 Frames, 159 Landmarks)
        # Note: Must match 'float32' as that is what TFLite expects
        dummy_input = np.zeros((1, 30, 159), dtype=np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print("✅ Verification passed! Model accepted input and produced output.")
        print("Output shape:", output_data.shape)
        
    except Exception as e:
        print(f"❌ Verification Failed: {e}")

if __name__ == "__main__":
    convert_model()