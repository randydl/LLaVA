from llava.train.train import train
from llava.train.patches import apply_compat_patches

if __name__ == "__main__":
    apply_compat_patches()
    train(attn_implementation="flash_attention_2")
