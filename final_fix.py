import sys
import os

# Add the fix for transformers import issue
import importlib.metadata

# Monkey patch importlib.metadata.version for problematic packages
original_version = importlib.metadata.version

def patched_version(package_name):
    # First try to get version from module directly for known problematic packages
    problematic_packages = ['torch', 'tokenizers']
    
    if package_name in problematic_packages:
        try:
            if package_name == 'torch':
                import torch
                if hasattr(torch, '__version__'):
                    return str(torch.__version__)
            elif package_name == 'tokenizers':
                import tokenizers
                if hasattr(tokenizers, '__version__'):
                    return tokenizers.__version__
        except ImportError:
            pass
    
    # Fall back to original
    result = original_version(package_name)
    
    # If still None, try module approach
    if result is None and package_name in problematic_packages:
        try:
            if package_name == 'torch':
                import torch
                if hasattr(torch, '__version__'):
                    return str(torch.__version__)
            elif package_name == 'tokenizers':
                import tokenizers
                if hasattr(tokenizers, '__version__'):
                    return tokenizers.__version__
        except ImportError:
            pass
    
    return result

importlib.metadata.version = patched_version

# Now test imports
print("Testing imports with patches...")
try:
    import transformers
    print(f"✓ transformers imported successfully (version: {transformers.__version__})")
    
    import peft
    print(f"✓ peft imported successfully (version: {peft.__version__})")
    
    from llava.model import LlavaLlamaForCausalLM
    print("✓ LlavaLlamaForCausalLM imported successfully")
    
    from llava.train.train_mem import train
    print("✓ llava.train.train_mem imported successfully")
    
    print("\n✅ All imports successful! Ready to run training.")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    
    # Try to work around EncoderDecoderCache issue
    print("\nAttempting to work around EncoderDecoderCache issue...")
    
    # Create a dummy EncoderDecoderCache class to fix the import
    import transformers
    if not hasattr(transformers, 'EncoderDecoderCache'):
        print("Adding dummy EncoderDecoderCache to transformers module...")
        from transformers.cache_utils import Cache
        transformers.EncoderDecoderCache = Cache
        transformers.DynamicCache = Cache
    
    # Try imports again
    try:
        from llava.train.train_mem import train
        print("✓ Successfully imported llava.train.train_mem after patch")
    except Exception as e2:
        print(f"✗ Still failed: {e2}")
