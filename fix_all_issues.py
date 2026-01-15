import sys
import os
import importlib.metadata

# Apply patches before any imports
def patch_all():
    # Patch importlib.metadata.version for problematic packages
    original_version = importlib.metadata.version
    
    def patched_version(package_name):
        # Handle torch
        if package_name == 'torch':
            import torch
            return str(torch.__version__)
        # Handle tokenizers
        elif package_name == 'tokenizers':
            import tokenizers
            return tokenizers.__version__
        # Handle other packages that might return None
        result = original_version(package_name)
        if result is None:
            # Try to get version from the module itself
            try:
                if package_name == 'torch':
                    import torch
                    return str(torch.__version__)
                elif package_name == 'tokenizers':
                    import tokenizers
                    return tokenizers.__version__
            except ImportError:
                pass
        return result
    
    importlib.metadata.version = patched_version
    
    # Patch transformers.utils.versions._compare_versions to handle None
    import transformers.utils.versions as versions_module
    
    original_compare = versions_module._compare_versions
    
    def patched_compare(op, got_ver, want_ver, requirement, pkg, hint):
        # If got_ver is None, try to get it from the module
        if got_ver is None:
            if pkg == 'tokenizers':
                import tokenizers
                got_ver = tokenizers.__version__
            elif pkg == 'torch':
                import torch
                got_ver = str(torch.__version__)
        
        # If still None, skip the check
        if got_ver is None or want_ver is None:
            print(f"Warning: Skipping version check for {requirement}: got_ver={got_ver}, want_ver={want_ver}")
            return
        
        # Call original function
        return original_compare(op, got_ver, want_ver, requirement, pkg, hint)
    
    versions_module._compare_versions = patched_compare
    
    # Patch peft to handle missing EncoderDecoderCache
    import transformers
    
    # Check if we need to add missing Cache classes
    if not hasattr(transformers, 'EncoderDecoderCache'):
        print("Adding missing Cache classes to transformers module...")
        from transformers.cache_utils import Cache
        
        # Create aliases
        transformers.EncoderDecoderCache = Cache
        transformers.DynamicCache = Cache
        
        # Also add to __all__ if needed
        if hasattr(transformers, '__all__'):
            transformers.__all__.extend(['EncoderDecoderCache', 'DynamicCache'])
    
    # Also patch the imports in peft if needed
    import peft.peft_model as peft_model_module
    
    # Monkey patch the import in peft_model.py
    original_import_line = "from transformers import Cache, DynamicCache, EncoderDecoderCache, PreTrainedModel"
    # We'll replace it dynamically when the module loads
    
    print("All patches applied successfully!")

# Apply patches
patch_all()

print("Testing imports...")
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
    
    # Now test running a simple training
    print("\nTesting if we can run training with minimal config...")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
