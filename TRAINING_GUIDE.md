# LLaVA 完整训练指南

## 训练流程概述

LLaVA 训练分为两个阶段：
1. **Pretrain（特征对齐阶段）**：使用 LAION-CC-SBU 数据集，连接冻结的视觉编码器和语言模型
2. **Finetune（视觉指令调优阶段）**：使用 LLaVA-Instruct-150K 等指令数据集，教模型遵循多模态指令

## 阶段一：Pretrain（特征对齐）

### 数据准备
1. 下载 LAION-CC-SBU 558K 数据集：
   ```bash
   # 数据集链接：https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
   ```

2. 数据集结构：
   ```
   /path/to/LLaVA-Pretrain/
   ├── blip_laion_cc_sbu_558k.json  # 标注文件
   └── images/                       # 图像文件
   ```

### 运行 Pretrain
```bash
# 使用自定义脚本
bash scripts/v1_5/pretrain_custom.sh

# 或手动运行
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /nas_train/app.e0031982/models/Qwen/Qwen3-4B-Instruct-2507 \
    --version plain \
    --data_path /nas_train/app.e0031982/datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /nas_train/app.e0031982/datasets/LLaVA-Pretrain \
    --vision_tower /nas_train/app.e0031982/models/google/siglip2-so400m-patch16-384 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-qwen3-4b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
```

### Pretrain 关键参数
- `--tune_mm_mlp_adapter True`：只训练多模态投影器
- `--learning_rate 1e-3`：较高的学习率
- `--model_max_length 2048`：最大序列长度
- `--vision_tower`：使用 SigLIP 视觉编码器

## 阶段二：Finetune（视觉指令调优）

### 数据准备
1. 下载 LLaVA-Instruct-150K 数据集：
   ```bash
   # 数据集链接：https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K
   ```

2. 下载其他视觉数据集：
   - COCO: train2017
   - GQA: images
   - OCR-VQA: images
   - TextVQA: train_images
   - VisualGenome: VG_100K, VG_100K_2

3. 组织数据目录：
   ```
   ./playground/data/
   ├── coco/train2017/
   ├── gqa/images/
   ├── ocr_vqa/images/
   ├── textvqa/train_images/
   └── vg/{VG_100K, VG_100K_2}/
   ```

### 运行 Finetune
```bash
# 使用自定义脚本
bash scripts/v1_5/finetune_custom.sh

# 或手动运行
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /nas_train/app.e0031982/models/Qwen/Qwen3-4B-Instruct-2507 \
    --version v1 \
    --data_path /nas_train/app.e0031982/datasets/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower /nas_train/app.e0031982/models/google/siglip2-so400m-patch16-384 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-qwen3-4b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-qwen3-4b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
```

### Finetune 关键参数
- `--pretrain_mm_mlp_adapter`：加载预训练的多模态投影器
- `--learning_rate 2e-5`：较低的学习率
- `--image_aspect_ratio pad`：填充非正方形图像（减少幻觉）
- `--group_by_modality_length True`：按模态长度分组，加速训练

## 硬件要求

### Pretrain 阶段
- **推荐**: 8x A100 80GB GPU
- **批次大小**: 32（每设备）
- **训练时间**: ~5.5 小时（LLaVA-v1.5-13B, 336px）

### Finetune 阶段
- **推荐**: 8x A100 80GB GPU
- **批次大小**: 16（每设备）
- **训练时间**: ~20 小时（LLaVA-v1.5-13B, 336px）

### 内存优化选项
1. **LoRA 训练**：减少 GPU 内存需求
   ```bash
   bash scripts/v1_5/finetune_lora.sh
   ```

2. **量化训练**：4-bit/8-bit 推理
   ```bash
   # 添加 --load-4bit 或 --load-8bit 参数
   ```

3. **CPU Offload**：使用 zero3_offload.json

## 环境修复

已修复以下导入兼容性问题：

### 1. importlib.metadata.version 返回 None
- **修复文件**: `transformers/utils/import_utils.py`
- **问题**: `importlib.metadata.version('torch')` 和 `importlib.metadata.version('tokenizers')` 返回 `None`
- **解决方案**: 直接导入包获取版本信息

### 2. transformers 模块缺少 DynamicCache 和 EncoderDecoderCache
- **修复文件**: `peft/peft_model.py`
- **问题**: peft 库尝试导入不存在的类
- **解决方案**: 添加兼容性导入处理

### 3. GradientCheckpointingLayer 导入失败
- **修复文件**: `peft/tuners/lora/model.py`
- **问题**: 从 `transformers.modeling_layers` 导入失败
- **解决方案**: 添加备用导入处理

## 验证环境

运行以下命令验证环境：
```bash
python verify_fixes.py
```

输出应显示：
```
✅ SUCCESS: All LLaVA imports work!
✅ All fixes applied successfully!
```

## 快速开始

### 1. 应用修复（如果需要）
```bash
python apply_fixes.py
```

### 2. 验证环境
```bash
python verify_fixes.py
```

### 3. 测试训练脚本
```bash
python test_training_simple.py
```

### 4. 运行 Pretrain
```bash
# 确保数据准备就绪
bash scripts/v1_5/pretrain_custom.sh
```

### 5. 运行 Finetune
```bash
# 确保 pretrain 检查点存在
bash scripts/v1_5/finetune_custom.sh
```

## 故障排除

### 常见问题
1. **CUDA 内存不足**：
   - 减少 `per_device_train_batch_size`
   - 增加 `gradient_accumulation_steps`
   - 使用 LoRA 训练

2. **导入错误**：
   - 运行 `python apply_fixes.py`
   - 检查 Python 环境是否正确

3. **数据加载错误**：
   - 确认数据路径正确
   - 检查 JSON 文件格式

4. **模型下载失败**：
   - 检查网络连接
   - 手动下载模型到指定路径

## 资源链接

- [LLaVA 官方 GitHub](https://github.com/haotian-liu/LLaVA)
- [LLaVA 模型 Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)
- [LLaVA 数据集](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)
- [自定义数据微调指南](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md)

---

**最后更新**: 2026-01-15  
**环境状态**: ✅ 修复完成，准备就绪  
**Git 提交**: 83015d8 (修复 LLaVA 训练环境导入问题)
