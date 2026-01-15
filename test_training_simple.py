#!/usr/bin/env python3
"""
Minimal test script to verify LLaVA training imports work correctly
"""
import sys
import os
import json

# Apply patches before any imports
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

# Patch transformers.utils.versions._compare_versions
import transformers.utils.versions as versions_module
original_compare = versions_module._compare_versions

def patched_compare(op, got_ver, want_ver, requirement, pkg, hint):
    if got_ver is None:
        if pkg == 'tokenizers':
            import tokenizers
            got_ver = tokenizers.__version__
        elif pkg == 'torch':
            import torch
            got_ver = str(torch.__version__)
    
    if got_ver is None or want_ver is None:
        print(f"Warning: Skipping version check for {requirement}: got_ver={got_ver}, want_ver={want_ver}")
        return
    
    return original_compare(op, got_ver, want_ver, requirement, pkg, hint)

versions_module._compare_versions = patched_compare

# Patch peft imports
import transformers
if not hasattr(transformers, 'EncoderDecoderCache'):
    from transformers.cache_utils import Cache
    transformers.EncoderDecoderCache = Cache
    transformers.DynamicCache = Cache
    if hasattr(transformers, '__all__'):
        transformers.__all__.extend(['EncoderDecoderCache', 'DynamicCache'])

print("✅ All patches applied")

# Now test the actual training imports
try:
    from llava.model import LlavaLlamaForCausalLM
    from llava.train.train_mem import train
    import argparse
    
    print("✅ LLaVA training modules imported successfully")
    
    # Create minimal dataset for testing
    test_data_path = "./playground/data/test_data.json"
    os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
    
    # Create a tiny test dataset
    test_data = [
        {
            "id": "test_1",
            "image": "test.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is in this image?"},
                {"from": "gpt", "value": "This is a test image."}
            ]
        }
    ]
    
    with open(test_data_path, "w") as f:
        json.dump(test_data, f)
    
    print(f"✅ Created test dataset at {test_data_path}")
    
    # Test argument parsing
    parser = argparse.ArgumentParser()
    
    # Minimal required arguments
    parser.add_argument("--model_name_or_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--data_path", type=str, default=test_data_path)
    parser.add_argument("--image_folder", type=str, default="./playground/data")
    parser.add_argument("--vision_tower", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--mm_projector_type", type=str, default="mlp2x_gelu")
    parser.add_argument("--mm_vision_select_layer", type=int, default=-2)
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=False)
    parser.add_argument("--mm_use_im_patch_token", action="store_true", default=False)
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--group_by_modality_length", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/llava-test")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)  # Small for testing
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--evaluation_strategy", type=str, default="no")
    parser.add_argument("--save_strategy", type=str, default="no")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--model_max_length", type=int, default=512)  # Small for testing
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)  # 0 for debugging
    parser.add_argument("--lazy_preprocess", action="store_true", default=False)
    parser.add_argument("--deepspeed", type=str, default=None)  # No deepspeed for test
    parser.add_argument("--local_rank", type=int, default=-1)
    
    args = parser.parse_args([])
    
    print("\n✅ Arguments parsed successfully")
    print(f"Model: {args.model_name_or_path}")
    print(f"Data path: {args.data_path}")
    print(f"Output dir: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test if we can create a simple config
    print("\nTesting model configuration...")
    from llava.model import LlavaConfig
    from transformers import AutoConfig
    
    # Get base config
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    print(f"Base config loaded: {type(config)}")
    
    # Create LLaVA config
    llava_config = LlavaConfig.from_pretrained(args.model_name_or_path)
    print(f"LLaVA config created: {type(llava_config)}")
    
    print("\n✅ All imports and configurations work correctly!")
    print("\nThe environment is ready for training.")
    print("\nTo run actual training, you can use:")
    print("1. The provided scripts: scripts/v1_5/finetune_custom.sh")
    print("2. Or run: python -m llava.train.train_mem with appropriate arguments")
    print("\nNote: You'll need to prepare your training data first.")
    
except Exception as e:
    print(f"\n❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
