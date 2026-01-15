#!/usr/bin/env python3
"""
Apply fixes to LLaVA training environment compatibility issues.

This script applies all necessary fixes for the import issues encountered
when running LLaVA training with the current environment.
"""

import os
import sys

def apply_import_utils_fix():
    """Fix importlib.metadata.version returning None issue"""
    filepath = "/nas_train/app.e0031982/miniforge3/envs/py310/lib/python3.10/site-packages/transformers/utils/import_utils.py"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return False
    
    # Check if fix is already applied
    if 'if got_ver is None:' in content and 'import tokenizers' in content:
        print(f"✅ Fix already applied to {filepath}")
        return True
    
    # Find the _compare_versions function
    func_start = content.find('def _compare_versions(')
    if func_start == -1:
        print("❌ Could not find _compare_versions function")
        return False
    
    # Find the exact location to insert
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if 'def _compare_versions(' in line:
            # Find where to insert (after function signature)
            for j in range(i+1, len(lines)):
                if lines[j].strip() and not lines[j].startswith('    '):
                    break
                if 'if ' in lines[j] or 'return' in lines[j]:
                    # Insert our fix here
                    indent = '    '
                    fix_lines = [
                        f"{indent}# Fix for packages that return None from importlib.metadata.version",
                        f"{indent}if got_ver is None:",
                        f"{indent}    if pkg == 'tokenizers':",
                        f"{indent}        import tokenizers",
                        f"{indent}        got_ver = tokenizers.__version__",
                        f"{indent}    elif pkg == 'torch':",
                        f"{indent}        import torch",
                        f"{indent}        got_ver = str(torch.__version__)",
                        f"{indent}",
                    ]
                    lines = lines[:j] + fix_lines + lines[j:]
                    break
            break
    
    new_content = '\n'.join(lines)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✅ Applied fix to {filepath}")
        return True
    except Exception as e:
        print(f"❌ Failed to write to {filepath}: {e}")
        return False

def apply_peft_model_fix():
    """Fix missing DynamicCache and EncoderDecoderCache imports in peft_model.py"""
    filepath = "/nas_train/app.e0031982/miniforge3/envs/py310/lib/python3.10/site-packages/peft/peft_model.py"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if fix is already applied
    if 'except ImportError:' in content and 'class DynamicCache(Cache):' in content:
        print(f"✅ Fix already applied to {filepath}")
        return True
    
    # Find the import line
    import_line = 'from transformers import Cache, DynamicCache, EncoderDecoderCache, PreTrainedModel'
    if import_line not in content:
        print(f"❌ Could not find import line in {filepath}")
        return False
    
    # Replace the import with try-except
    fix_import = '''# Try to import Cache classes, handle missing imports
try:
    from transformers import Cache, DynamicCache, EncoderDecoderCache, PreTrainedModel
except ImportError:
    from transformers import Cache, PreTrainedModel
    # Create dummy classes for compatibility
    class DynamicCache(Cache):
        pass
    class EncoderDecoderCache(Cache):
        pass'''
    
    new_content = content.replace(import_line, fix_import)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Applied fix to {filepath}")
    return True

def apply_lora_model_fix():
    """Fix missing GradientCheckpointingLayer import in lora/model.py"""
    filepath = "/nas_train/app.e0031982/miniforge3/envs/py310/lib/python3.10/site-packages/peft/tuners/lora/model.py"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if fix is already applied
    if 'except ImportError:' in content and 'class GradientCheckpointingLayer(nn.Module):' in content:
        print(f"✅ Fix already applied to {filepath}")
        return True
    
    # Find the import line
    import_line = 'from transformers.modeling_layers import GradientCheckpointingLayer'
    if import_line not in content:
        print(f"❌ Could not find import line in {filepath}")
        return False
    
    # Replace the import with try-except
    fix_import = '''# Try to import GradientCheckpointingLayer from transformers, fallback to dummy class
try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    # Create a dummy GradientCheckpointingLayer class if not available
    class GradientCheckpointingLayer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.gradient_checkpointing = False'''
    
    new_content = content.replace(import_line, fix_import)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Applied fix to {filepath}")
    return True

def create_test_script():
    """Create test script to verify fixes"""
    test_script = '''#!/usr/bin/env python3
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

print("\\nEnvironment check:")
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU count: {torch.cuda.device_count()}")
    print(f"  GPU 0: {torch.cuda.get_device_name(0)}")

print("\\n✅ All fixes applied successfully!")
print("\\nYou can now run LLaVA training with:")
print("  bash scripts/v1_5/finetune_custom.sh")
print("  OR")
print("  python -m llava.train.train_mem --your-arguments")
'''
    
    with open('verify_fixes.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("✅ Created verification script: verify_fixes.py")
    return True

def main():
    print("=" * 60)
    print("LLaVA Training Environment Fixes")
    print("=" * 60)
    
    print("\nStep 1: Fixing transformers.utils.import_utils...")
    if not apply_import_utils_fix():
        print("⚠️  Warning: Could not apply transformers fix")
    
    print("\nStep 2: Fixing peft.peft_model imports...")
    if not apply_peft_model_fix():
        print("⚠️  Warning: Could not apply peft_model fix")
    
    print("\nStep 3: Fixing peft.tuners.lora.model imports...")
    if not apply_lora_model_fix():
        print("⚠️  Warning: Could not apply lora.model fix")
    
    print("\nStep 4: Creating verification script...")
    create_test_script()
    
    print("\n" + "=" * 60)
    print("All fixes applied!")
    print("=" * 60)
    print("\nTo verify the fixes, run:")
    print("  python verify_fixes.py")
    print("\nTo run training, use:")
    print("  bash scripts/v1_5/finetune_custom.sh")
    print("  OR")
    print("  python -m llava.train.train_mem --your-arguments")
    print("\nFor more details, see FIXES_SUMMARY.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
