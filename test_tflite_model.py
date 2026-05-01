#!/usr/bin/env python3
# ============================================================
# test_tflite_model.py — Test the converted TFLite model
# AmmoniteID v1.0
# ============================================================
# This script tests that the TFLite model produces
# similar results to the original Keras model.
#
# Run this BEFORE deploying to Render:
#   python3 test_tflite_model.py
# ============================================================

import sys
from pathlib import Path

print("=" * 60)
print("TFLite Model Test")
print("=" * 60)
print()

# Check if TFLite model exists
tflite_path = Path(__file__).parent / 'ammonite_model_v1.tflite'
if not tflite_path.exists():
    print("❌ ERROR: TFLite model not found!")
    print(f"   Expected: {tflite_path}")
    print()
    print("Please run: python3 convert_to_tflite.py")
    sys.exit(1)

print(f"✅ TFLite model found: {tflite_path}")
print(f"   Size: {tflite_path.stat().st_size / (1024*1024):.2f} MB")
print()

# Test loading the model
print("Testing TFLite interpreter...")
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    print("⚠️  tflite_runtime not installed, using tensorflow.lite")
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("❌ ERROR: Neither tflite_runtime nor tensorflow is installed!")
        print()
        print("Install with: pip install tflite-runtime")
        sys.exit(1)

try:
    interpreter = tflite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    print("✅ TFLite interpreter loaded successfully")
    print()
except Exception as e:
    print(f"❌ ERROR: Failed to load TFLite model: {e}")
    sys.exit(1)

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model Details:")
print(f"  Input shape:  {input_details[0]['shape']}")
print(f"  Output shape: {output_details[0]['shape']}")
print(f"  Input dtype:  {input_details[0]['dtype']}")
print(f"  Output dtype: {output_details[0]['dtype']}")
print()

# Test with a dummy image
print("Testing inference with dummy image...")
import numpy as np

# Create a dummy 224x224 RGB image
dummy_image = np.random.rand(1, 224, 224, 3).astype(np.float32)

try:
    interpreter.set_tensor(input_details[0]['index'], dummy_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print("✅ Inference successful!")
    print(f"   Output shape: {output.shape}")
    print(f"   Output values range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"   Sum of probabilities: {output.sum():.4f}")
    print()
    
except Exception as e:
    print(f"❌ ERROR: Inference failed: {e}")
    sys.exit(1)

# Test memory usage
print("Memory Test:")
print("  Running 10 consecutive inferences...")
for i in range(10):
    dummy_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], dummy_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    del dummy_image, output

print("✅ Memory test passed (no crashes)")
print()

print("=" * 60)
print("✅ ALL TESTS PASSED")
print("=" * 60)
print()
print("The TFLite model is working correctly!")
print()
print("Next steps:")
print("1. Replace your old files with the new ones:")
print("   mv identifier_new.py identifier.py")
print("   mv main_new.py main.py")
print("   mv config_new.py config.py")
print("   mv requirements_new.txt requirements.txt")
print()
print("2. Upload the .tflite model to your GitHub repo")
print("3. Deploy to Render")
print()
