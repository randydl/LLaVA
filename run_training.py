#!/usr/bin/env python3
import sys
import os
import importlib.metadata

# Apply patches before any imports
print("Applying compatibility patches...")

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

# Patch peft to handle missing imports
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

print("Patches applied successfully!")

# Now run the training script
print("\nStarting training...")
print("=" * 60)

# Import the training module
from llava.train.train_mem import train

# Run training with minimal config for testing
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--data_path", type=str, default="./playground/data")
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--vision_tower", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--mm_vision_select_layer", type=int, default=-2)
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/llava-v1.5-7b-finetune")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--evaluation_strategy", type=str, default="no")
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--deepspeed", type=str, default="./scripts/zero3.json")
    parser.add_argument("--local_rank", type=int, default=None)
    
    args = parser.parse_args([])  # Empty args for testing
    
    print(f"Training configuration:")
    print(f"  Model: {args.model_name_or_path}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Batch size: {args.per_device_train_batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  BF16: {args.bf16}")
    print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"  DeepSpeed: {args.deepspeed}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training
    try:
        train(parser)
        print("\n✅ Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
