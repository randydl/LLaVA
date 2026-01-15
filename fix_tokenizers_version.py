import sys
import importlib.metadata

# Monkey patch importlib.metadata.version for tokenizers
original_version = importlib.metadata.version

def patched_version(package_name):
    if package_name == 'tokenizers':
        # Try to get version from the module directly
        try:
            import tokenizers
            if hasattr(tokenizers, '__version__'):
                return tokenizers.__version__
        except ImportError:
            pass
    # Fall back to original
    result = original_version(package_name)
    if result is None and package_name == 'tokenizers':
        # If still None, try to get it from the module
        try:
            import tokenizers
            if hasattr(tokenizers, '__version__'):
                return tokenizers.__version__
        except ImportError:
            pass
    return result

importlib.metadata.version = patched_version

# Now test
print("Testing tokenizers version detection...")
try:
    ver = importlib.metadata.version('tokenizers')
    print(f"Tokenizers version (patched): {ver}")
    
    # Now try to import transformers
    import transformers
    print("Transformers import successful!")
    
    # Test LLaVA import
    from llava.model import LlavaLlamaForCausalLM
    print("LlavaLlamaForCausalLM import successful!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
