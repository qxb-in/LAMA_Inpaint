import os
import torch
from diffusers import AutoPipelineForInpainting
from optimum.exporters.onnx import export_model

# 加载 inpainting pipeline
pipe = AutoPipelineForInpainting.from_pretrained("dreamshaper-8-inpainting", torch_dtype=torch.float16, variant="fp16")

