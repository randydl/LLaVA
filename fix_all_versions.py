import sys
import importlib.metadata
import importlib.util

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

# Also patch accelerate.utils.versions
import accelerate.utils.versions as accelerate_versions
original_parse = accelerate_versions.parse

def patched_parse_versions(version_str):
    # If version_str is None, return a dummy version
    if version_str is None:
        from packaging.version import Version
        return Version("0.0.0")
    return original_parse(version_str)

accelerate_versions.parse = patched_parse_versions

print("Testing imports...")
try:
    # Test torch version
    torch_ver = importlib.metadata.version('torch')
    print(f"Torch version: {torch_ver}")
    
    # Test tokenizers version
    tokenizers_ver = importlib.metadata.version('tokenizers')
    print(f"Tokenizers version: {tokenizers_ver}")
    
    # Now try to import transformers
    import transformers
    print("Transformers import successful!")
    
    # Test LLaVA import
    from llava.model import LlavaLlamaForCausalLM
    print("LlavaLlamaForCausalLM import successful!")
    
    print("\nAll imports successful! Ready to run training.")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
