import torch
import transformers
from packaging import version

def apply_compat_patches():
    """应用新版PyTorch/Transformers兼容性补丁"""
    
    # 1. 处理flash attention兼容性
    if version.parse(torch.__version__) >= version.parse("2.1.0"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # 2. 修复transformers Trainer兼容性
    if not hasattr(transformers.Trainer, 'accumulate_grad_batches'):
        transformers.Trainer.accumulate_grad_batches = lambda self: self.args.gradient_accumulation_steps
