
from diffusers import AutoPipelineForInpainting, DEISMultistepScheduler
import torch
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import imageio

def extract_brush_strokes(image_dict):
    # 提取用户涂抹的图层
    layers = image_dict['layers'][0]  # 假设只有一个图层，取第一个

    alpha_channel = layers[:, :, 3]  # 获取图层的Alpha通道
    # 创建一个mask模板，标记非透明的区域（即用户涂抹的区域）
    mask = alpha_channel > 0  # True表示用户涂抹的区域
    # 将蒙版转换为图像形式
    mask_img = Image.fromarray(np.uint8(mask) * 255, mode="L")  # 转换为灰度图像
    mask_img.save('./data/data_mask.jpg')


pipe = AutoPipelineForInpainting.from_pretrained('dreamshaper-8-inpainting', torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda:7")

def predict(img_dict, prompt):
    # 提取涂抹图层
    extract_brush_strokes(img_dict)
    img = img_dict['background']
    
    image_rgb = Image.fromarray(img).convert("RGB")
    image_path = "./data/data.jpg"
    imageio.imwrite(image_path, np.array(image_rgb))

    img = load_image(image_path)
    w,h = img.size
    mask_img = load_image('./data/data_mask.jpg')

    # 随机数种子
    # generator = torch.manual_seed(33)
    image = pipe(prompt, image=img,  mask_image=mask_img, num_inference_steps=25).images[0]  
    w1,h1 = image.size
    image = image.resize((int(w*h1/h),h1))
    image.save("./dataout/result.jpg")

    return "./dataout/result.jpg", './data/data_mask.jpg'

import gradio as gr

if __name__=="__main__":

    with gr.Blocks() as iface:
        gr.Markdown("# Knowdee Image Inpainting")

        input_image = gr.ImageEditor(type="numpy")
        with gr.Row():
            prompt = gr.Textbox(label="prompt")
            btn_infer = gr.Button("Run")
            
        output_inpainted = gr.Image(type="filepath", label="Inpainted Image")

        with gr.Row():
            output_mask = gr.Image(type="filepath", label="Generated Mask")

        btn_infer.click(fn=predict, inputs=[input_image, prompt], outputs=[output_inpainted, output_mask])

    iface.launch(server_name="0.0.0.0", server_port=8067)