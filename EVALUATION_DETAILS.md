# LLaVA Pretrain 后模型的 Image Caption 评估指南

## 评估准备

### 1. 安装 LMMs-Eval
```bash
pip install lmms-eval
```

### 2. 准备评估数据
推荐使用以下数据集评估 image caption 能力：
- **COCO Captions** (标准评估数据集)
- **NoCaps** (开放域描述)
- **Flickr30k** (多参考评估)
- **TextCaps** (基于文本的图像描述)

## 适合 Pretrain 模型的评估任务

### 1. COCO Captioning
```python
from lmms_eval.tasks import coco_caption

eval_cfg = {
    "model": "pretrained_model_path",
    "device": "cuda",
    "batch_size": 8,
    "num_beams": 5
}

results = coco_caption.evaluate(eval_cfg)
print("CIDEr:", results["CIDEr"])
print("BLEU-4:", results["BLEU-4"])
```

### 2. NoCaps Challenge
```python
from lmms_eval.tasks import nocaps

results = nocaps.evaluate("pretrained_model_path", split="val")
print("CIDEr:", results["CIDEr"])
print("SPICE:", results["SPICE"])
```

### 3. 跨模态检索评估
评估视觉-语言对齐能力：
```python
from lmms_eval.tasks import cross_modal_retrieval

# 图像到文本检索
results = cross_modal_retrieval.evaluate(
    model_path="pretrained_model_path",
    task="image_to_text"
)
```

## 关键评估指标

1. **CIDEr** (Consensus-based Image Description Evaluation):
   - 最常用的 caption 评估指标
   - 衡量生成描述与人类参考的相似度
   - 对 pretrain 模型的描述准确性很敏感

2. **BLEU-4**:
   - 衡量 n-gram 重叠
   - 适用于基础 caption 能力评估

3. **SPICE**:
   - 评估语义内容匹配
   - 关注描述中的对象、属性和关系

## 评估脚本示例

### COCO Caption 评估完整示例
```python
import torch
from transformers import AutoModelForVision2Seq
from lmms_eval.tasks import coco_caption

# 加载 pretrain 模型
model = AutoModelForVision2Seq.from_pretrained(
    "path/to/your/pretrained/model",
    torch_dtype=torch.bfloat16
).to("cuda")

# 评估配置
eval_cfg = {
    "model": model,
    "processor": "pretrained_processor",
    "device": "cuda",
    "batch_size": 8,
    "gen_kwargs": {"max_length": 50, "num_beams": 5}
}

# 运行评估
results = coco_caption.evaluate(eval_cfg)

# 打印结果
print(f"""
COCO Caption 评估结果:
- CIDEr: {results['CIDEr']:.2f}
- BLEU-4: {results['BLEU-4']:.2f}
- METEOR: {results['METEOR']:.2f}
- SPICE: {results['SPICE']:.2f}
""")
```

## 评估建议

1. **pretrain 模型评估重点**:
   - 关注视觉概念的基础理解能力
   - 检查模型对常见物体的识别准确性
   - 评估描述的多样性

2. **对比评估**:
   - 与 baseline 模型 (如 BLIP, BLIP2) 比较
   - 跟踪不同 pretrain epoch 的性能变化

3. **人工评估** (重要):
   ```python
   from lmms_eval.tasks import human_eval

   samples = human_eval.generate_samples(
       model="pretrained_model_path",
       dataset="coco_val",
       num_samples=100
   )
   human_eval.save_for_rating(samples, "eval_results.json")
   ```

## 评估结果记录

建议创建评估记录表：

| 模型版本 | 数据集 | CIDEr | BLEU-4 | SPICE | 评估日期 |
|----------|--------|-------|--------|-------|----------|
| v1-pretrain | COCO | 85.2 | 32.1 | 15.6 | 2026-01-15 |
| v1-pretrain | NoCaps | 72.3 | 28.4 | 13.2 | 2026-01-15 |

## 高级评估技巧

1. **基于提示的评估**:
   ```python
   from lmms_eval.tasks import prompt_based_eval

   results = prompt_based_eval(
       model="pretrained_model_path",
       prompts=["描述这张图片", "这张图片里有什么"],
       dataset="coco_val"
   )
   ```

2. **稳健性评估**:
   - 添加噪声评估
   - 遮挡测试
   - 跨域评估 (如卡通图像)

## 资源链接
- [LMMs-Eval 官方文档](https://lmms-lab.github.io/lmms-eval/)
- [COCO 评估工具](https://github.com/tylin/coco-caption)
- [NoCaps 挑战](https://nocaps.org/)
