import torch
from torchvision.models import vit_b_16

# 加载预训练的Vision Transformer模型
model = vit_b_16(pretrained=True)