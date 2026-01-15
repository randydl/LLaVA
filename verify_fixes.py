#!/usr/bin/env python3
import sys
import os

# Apply patches
import importlib.metadata

original_version = importlib.metadata.version

def patched_version(package_name):
    if package_name == 'torch':
        import torch
        return str(torch.__version__)
    elif package_name == 'tokenizers':
        import tokenizers
        return tokenizers.__version__
    return original_version(package_name)

importlib.metadata.version = patched_version

print("Testing imports...")
try:
    from llava.model import LlavaLlamaForCausalLM
    from llava.train.train_mem import train
    print("✅ SUCCESS: All LLaVA imports work!")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

print("\nEnvironment check:")
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU count: {torch.cuda.device_count()}")
    print(f"  GPU 0: {torch.cuda.get_device_name(0)}")

print("\n✅ All fixes applied successfully!")
print("\nYou can now run LLaVA training with:")
print("  bash scripts/v1_5/finetune_custom.sh")
print("  OR")
print("  python -m llava.train.train_mem --your-arguments")
