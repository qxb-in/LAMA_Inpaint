
from diffusers import AutoPipelineForInpainting, DEISMultistepScheduler
import torch
from diffusers.utils import load_image
from PIL import Image
import requests
import numpy as np
import imageio

def box(x1, y1, x2, y2, path):

    data = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "path": path,
    }
    url = "http://:8005/box"

    res = requests.post(url, json=data)
    
    with open('./data/data_mask_by_box.jpg', "wb") as f:
        f.write(res.content)
    return res.content

def extract_brush_strokes(image_dict):
    # 提取用户涂抹的图层
    layers = image_dict['layers'][0]  # 假设只有一个图层，取第一个

    alpha_channel = layers[:, :, 3]  # 获取图层的Alpha通道
    # 创建一个mask模板，标记非透明的区域（即用户涂抹的区域）
    mask = alpha_channel > 0  # True表示用户涂抹的区域
    # 将蒙版转换为图像形式
    mask_img = Image.fromarray(np.uint8(mask) * 255, mode="L")  # 转换为灰度图像
    mask_img.save('./data/data_mask.jpg')

def get_bounding_box(image: np.ndarray):
    """
    获取涂抹区域的最小边界框(x_min, y_min, x_max, y_max)，分别表示边界框的左上角和右下角的坐标。
    """
    # 假设涂抹区域是非零值，可以是二值图像或其他格式
    # 对于RGBA图像，可以取alpha通道来判断哪些区域被涂抹
    if image.shape[-1] == 4:  # 如果是RGBA图像
        smudge_mask = image[..., 3] > 0  # 获取Alpha通道非透明区域
    else:
        smudge_mask = image > 0  # 假设其他通道非零值表示涂抹

    # 获取所有涂抹区域的坐标点
    coords = np.argwhere(smudge_mask)

    if coords.shape[0] == 0:
        raise ValueError("没有找到涂抹区域")

    # 分别计算最小和最大坐标
    y_min, x_min = coords.min(axis=0)  # 找到最小的x, y
    y_max, x_max = coords.max(axis=0)  # 找到最大的x, y

    return int(x_min), int(y_min), int(x_max), int(y_max)

#pipe = AutoPipelineForInpainting.from_pretrained('dreamshaper-8-inpainting', torch_dtype=torch.float16, variant="fp16")
pipe = AutoPipelineForInpainting.from_pretrained('dreamshaper-8-inpainting', variant="fp16")
pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
#pipe = pipe.to("cuda:1")
pipe = pipe.to("cpu")

css = """
:root {
  --name: default;
  --primary-50: #fff7ed;
  --primary-100: #ffedd5;
  --primary-200: #fed7aa;
  --primary-300: #fdba74;
  --primary-400: #fb923c;
  --primary-500: #f97316;
  --primary-600: #ea580c;
  --primary-700: #c2410c;
  --primary-800: #9a3412;
  --primary-900: #7c2d12;
  --primary-950: #6c2e12;
  --secondary-50: #eff6ff;
  --secondary-100: #dbeafe;
  --secondary-200: #bfdbfe;
  --secondary-300: #93c5fd;
  --secondary-400: #60a5fa;
  --secondary-500: #3b82f6;
  --secondary-600: #2563eb;
  --secondary-700: #1d4ed8;
  --secondary-800: #1e40af;
  --secondary-900: #1e3a8a;
  --secondary-950: #1d3660;
  --neutral-50: #f9fafb;
  --neutral-100: #f3f4f6;
  --neutral-200: #e5e7eb;
  --neutral-300: #d1d5db;
  --neutral-400: #9ca3af;
  --neutral-500: #6b7280;
  --neutral-600: #4b5563;
  --neutral-700: #374151;
  --neutral-800: #1f2937;
  --neutral-900: #111827;
  --neutral-950: #0b0f19;
  --spacing-xxs: 1px;
  --spacing-xs: 2px;
  --spacing-sm: 4px;
  --spacing-md: 6px;
  --spacing-lg: 8px;
  --spacing-xl: 10px;
  --spacing-xxl: 16px;
  --radius-xxs: 1px;
  --radius-xs: 2px;
  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
  --radius-xl: 12px;
  --radius-xxl: 22px;
  --text-xxs: 9px;
  --text-xs: 10px;
  --text-sm: 12px;
  --text-md: 14px;
  --text-lg: 16px;
  --text-xl: 22px;
  --text-xxl: 26px;
  --font: 'Source Sans Pro', 'ui-sans-serif', 'system-ui', sans-serif;
  --font-mono: 'IBM Plex Mono', 'ui-monospace', 'Consolas', monospace;
  --body-background-fill: var(--background-fill-primary);
  --body-text-color: var(--neutral-800);
  --body-text-size: var(--text-md);
  --body-text-weight: 400;
  --embed-radius: var(--radius-lg);
  --color-accent: var(--primary-500);
  --color-accent-soft: var(--primary-50);
  --background-fill-primary: white;
  --background-fill-secondary: var(--neutral-50);
  --border-color-accent: var(--primary-300);
  --border-color-primary: var(--neutral-200);
  --link-text-color: var(--secondary-600);
  --link-text-color-active: var(--secondary-600);
  --link-text-color-hover: var(--secondary-700);
  --link-text-color-visited: var(--secondary-500);
  --body-text-color-subdued: var(--neutral-400);
  --accordion-text-color: var(--body-text-color);
  --table-text-color: var(--body-text-color);
  --shadow-drop: rgba(0,0,0,0.05) 0px 1px 2px 0px;
  --shadow-drop-lg: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
  --shadow-inset: rgba(0,0,0,0.05) 0px 2px 4px 0px inset;
  --shadow-spread: 3px;
  --block-background-fill: var(--background-fill-primary);
  --block-border-color: var(--border-color-primary);
  --block-border-width: 1px;
  --block-info-text-color: var(--body-text-color-subdued);
  --block-info-text-size: var(--text-sm);
  --block-info-text-weight: 400;
  --block-label-background-fill: var(--background-fill-primary);
  --block-label-border-color: var(--border-color-primary);
  --block-label-border-width: 1px;
  --block-label-shadow: var(--block-shadow);
  --block-label-text-color: var(--neutral-500);
  --block-label-margin: 0;
  --block-label-padding: var(--spacing-sm) var(--spacing-lg);
  --block-label-radius: calc(var(--radius-lg) - 1px) 0 calc(var(--radius-lg) - 1px) 0;
  --block-label-right-radius: 0 calc(var(--radius-lg) - 1px) 0 calc(var(--radius-lg) - 1px);
  --block-label-text-size: var(--text-sm);
  --block-label-text-weight: 400;
  --block-padding: var(--spacing-xl) calc(var(--spacing-xl) + 2px);
  --block-radius: var(--radius-lg);
  --block-shadow: var(--shadow-drop);
  --block-title-background-fill: none;
  --block-title-border-color: none;
  --block-title-border-width: 0px;
  --block-title-text-color: var(--neutral-500);
  --block-title-padding: 0;
  --block-title-radius: none;
  --block-title-text-size: var(--text-md);
  --block-title-text-weight: 400;
  --container-radius: var(--radius-lg);
  --form-gap-width: 1px;
  --layout-gap: var(--spacing-xxl);
  --panel-background-fill: var(--background-fill-secondary);
  --panel-border-color: var(--border-color-primary);
  --panel-border-width: 0;
  --section-header-text-size: var(--text-md);
  --section-header-text-weight: 400;
  --border-color-accent-subdued: var(--primary-200);
  --code-background-fill: var(--neutral-100);
  --checkbox-background-color: var(--background-fill-primary);
  --checkbox-background-color-focus: var(--checkbox-background-color);
  --checkbox-background-color-hover: var(--checkbox-background-color);
  --checkbox-background-color-selected: var(--secondary-600);
  --checkbox-border-color: var(--neutral-300);
  --checkbox-border-color-focus: var(--secondary-500);
  --checkbox-border-color-hover: var(--neutral-300);
  --checkbox-border-color-selected: var(--secondary-600);
  --checkbox-border-radius: var(--radius-sm);
  --checkbox-border-width: var(--input-border-width);
  --checkbox-label-background-fill: linear-gradient(to top, var(--neutral-50), white);
  --checkbox-label-background-fill-hover: linear-gradient(to top, var(--neutral-100), white);
  --checkbox-label-background-fill-selected: var(--checkbox-label-background-fill);
  --checkbox-label-border-color: var(--border-color-primary);
  --checkbox-label-border-color-hover: var(--checkbox-label-border-color);
  --checkbox-label-border-width: var(--input-border-width);
  --checkbox-label-gap: var(--spacing-lg);
  --checkbox-label-padding: var(--spacing-md) calc(2 * var(--spacing-md));
  --checkbox-label-shadow: var(--shadow-drop);
  --checkbox-label-text-size: var(--text-md);
  --checkbox-label-text-weight: 400;
  --checkbox-check: url("data:image/svg+xml,%3csvg viewBox='0 0 16 16' fill='white' xmlns='http://www.w3.org/2000/svg'%3e%3cpath d='M12.207 4.793a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0l-2-2a1 1 0 011.414-1.414L6.5 9.086l4.293-4.293a1 1 0 011.414 0z'/%3e%3c/svg%3e");
  --radio-circle: url("data:image/svg+xml,%3csvg viewBox='0 0 16 16' fill='white' xmlns='http://www.w3.org/2000/svg'%3e%3ccircle cx='8' cy='8' r='3'/%3e%3c/svg%3e");
  --checkbox-shadow: var(--input-shadow);
  --checkbox-label-text-color: var(--body-text-color);
  --checkbox-label-text-color-selected: var(--checkbox-label-text-color);
  --error-background-fill: #fef2f2;
  --error-border-color: #b91c1c;
  --error-border-width: 1px;
  --error-text-color: #b91c1c;
  --error-icon-color: #b91c1c;
  --input-background-fill: white;
  --input-background-fill-focus: var(--secondary-500);
  --input-background-fill-hover: var(--input-background-fill);
  --input-border-color: var(--border-color-primary);
  --input-border-color-focus: var(--secondary-300);
  --input-border-color-hover: var(--input-border-color);
  --input-border-width: 1px;
  --input-padding: var(--spacing-xl);
  --input-placeholder-color: var(--neutral-400);
  --input-radius: var(--radius-lg);
  --input-shadow: 0 0 0 var(--shadow-spread) transparent, var(--shadow-inset);
  --input-shadow-focus: 0 0 0 var(--shadow-spread) var(--secondary-50), var(--shadow-inset);
  --input-text-size: var(--text-md);
  --input-text-weight: 400;
  --loader-color: var(--color-accent);
  --prose-text-size: var(--text-md);
  --prose-text-weight: 400;
  --prose-header-text-weight: 600;
  --slider-color: #2563eb;
  --stat-background-fill: linear-gradient(to right, var(--primary-400), var(--primary-200));
  --table-border-color: var(--neutral-300);
  --table-even-background-fill: white;
  --table-odd-background-fill: var(--neutral-50);
  --table-radius: var(--radius-lg);
  --table-row-focus: var(--color-accent-soft);
  --button-border-width: var(--input-border-width);
  --button-cancel-background-fill: linear-gradient(to bottom right, #fee2e2, #fecaca);
  --button-cancel-background-fill-hover: linear-gradient(to bottom right, #fee2e2, #fee2e2);
  --button-cancel-border-color: #fecaca;
  --button-cancel-border-color-hover: var(--button-cancel-border-color);
  --button-cancel-text-color: #dc2626;
  --button-cancel-text-color-hover: var(--button-cancel-text-color);
  --button-large-padding: var(--spacing-lg) calc(2 * var(--spacing-lg));
  --button-large-radius: var(--radius-lg);
  --button-large-text-size: var(--text-lg);
  --button-large-text-weight: 600;
  --button-primary-background-fill: linear-gradient(to bottom right, var(--primary-100), var(--primary-300));
  --button-primary-background-fill-hover: linear-gradient(to bottom right, var(--primary-100), var(--primary-200));
  --button-primary-border-color: var(--primary-200);
  --button-primary-border-color-hover: var(--button-primary-border-color);
  --button-primary-text-color: var(--primary-600);
  --button-primary-text-color-hover: var(--button-primary-text-color);
  --button-secondary-background-fill: linear-gradient(to bottom right, var(--neutral-100), var(--neutral-200));
  --button-secondary-background-fill-hover: linear-gradient(to bottom right, var(--neutral-100), var(--neutral-100));
  --button-secondary-border-color: var(--neutral-200);
  --button-secondary-border-color-hover: var(--button-secondary-border-color);
  --button-secondary-text-color: var(--neutral-700);
  --button-secondary-text-color-hover: var(--button-secondary-text-color);
  --button-shadow: var(--shadow-drop);
  --button-shadow-active: var(--shadow-inset);
  --button-shadow-hover: var(--shadow-drop-lg);
  --button-small-padding: var(--spacing-sm) calc(2 * var(--spacing-sm));
  --button-small-radius: var(--radius-lg);
  --button-small-text-size: var(--text-md);
  --button-small-text-weight: 400;
  --button-transition: none;
}
.dark {
  --body-background-fill: var(--background-fill-primary);
  --body-text-color: var(--neutral-100);
  --color-accent-soft: var(--neutral-700);
  --background-fill-primary: var(--neutral-950);
  --background-fill-secondary: var(--neutral-900);
  --border-color-accent: var(--neutral-600);
  --border-color-primary: var(--neutral-700);
  --link-text-color-active: var(--secondary-500);
  --link-text-color: var(--secondary-500);
  --link-text-color-hover: var(--secondary-400);
  --link-text-color-visited: var(--secondary-600);
  --body-text-color-subdued: var(--neutral-400);
  --accordion-text-color: var(--body-text-color);
  --table-text-color: var(--body-text-color);
  --shadow-spread: 1px;
  --block-background-fill: var(--neutral-800);
  --block-border-color: var(--border-color-primary);
  --block_border_width: None;
  --block-info-text-color: var(--body-text-color-subdued);
  --block-label-background-fill: var(--background-fill-secondary);
  --block-label-border-color: var(--border-color-primary);
  --block_label_border_width: None;
  --block-label-text-color: var(--neutral-200);
  --block_shadow: None;
  --block_title_background_fill: None;
  --block_title_border_color: None;
  --block_title_border_width: None;
  --block-title-text-color: var(--neutral-200);
  --panel-background-fill: var(--background-fill-secondary);
  --panel-border-color: var(--border-color-primary);
  --panel_border_width: None;
  --border-color-accent-subdued: var(--border-color-accent);
  --code-background-fill: var(--neutral-800);
  --checkbox-background-color: var(--neutral-800);
  --checkbox-background-color-focus: var(--checkbox-background-color);
  --checkbox-background-color-hover: var(--checkbox-background-color);
  --checkbox-background-color-selected: var(--secondary-600);
  --checkbox-border-color: var(--neutral-700);
  --checkbox-border-color-focus: var(--secondary-500);
  --checkbox-border-color-hover: var(--neutral-600);
  --checkbox-border-color-selected: var(--secondary-600);
  --checkbox-border-width: var(--input-border-width);
  --checkbox-label-background-fill: linear-gradient(to top, var(--neutral-900), var(--neutral-800));
  --checkbox-label-background-fill-hover: linear-gradient(to top, var(--neutral-900), var(--neutral-800));
  --checkbox-label-background-fill-selected: var(--checkbox-label-background-fill);
  --checkbox-label-border-color: var(--border-color-primary);
  --checkbox-label-border-color-hover: var(--checkbox-label-border-color);
  --checkbox-label-border-width: var(--input-border-width);
  --checkbox-label-text-color: var(--body-text-color);
  --checkbox-label-text-color-selected: var(--checkbox-label-text-color);
  --error-background-fill: var(--neutral-900);
  --error-border-color: #ef4444;
  --error_border_width: None;
  --error-text-color: #fef2f2;
  --error-icon-color: #ef4444;
  --input-background-fill: var(--neutral-800);
  --input-background-fill-focus: var(--secondary-600);
  --input-background-fill-hover: var(--input-background-fill);
  --input-border-color: var(--border-color-primary);
  --input-border-color-focus: var(--neutral-700);
  --input-border-color-hover: var(--input-border-color);
  --input_border_width: None;
  --input-placeholder-color: var(--neutral-500);
  --input_shadow: None;
  --input-shadow-focus: 0 0 0 var(--shadow-spread) var(--neutral-700), var(--shadow-inset);
  --loader_color: None;
  --slider_color: None;
  --stat-background-fill: linear-gradient(to right, var(--primary-400), var(--primary-600));
  --table-border-color: var(--neutral-700);
  --table-even-background-fill: var(--neutral-950);
  --table-odd-background-fill: var(--neutral-900);
  --table-row-focus: var(--color-accent-soft);
  --button-border-width: var(--input-border-width);
  --button-cancel-background-fill: linear-gradient(to bottom right, #dc2626, #b91c1c);
  --button-cancel-background-fill-hover: linear-gradient(to bottom right, #dc2626, #dc2626);
  --button-cancel-border-color: #dc2626;
  --button-cancel-border-color-hover: var(--button-cancel-border-color);
  --button-cancel-text-color: white;
  --button-cancel-text-color-hover: var(--button-cancel-text-color);
  --button-primary-background-fill: linear-gradient(to bottom right, var(--primary-500), var(--primary-600));
  --button-primary-background-fill-hover: linear-gradient(to bottom right, var(--primary-500), var(--primary-500));
  --button-primary-border-color: var(--primary-500);
  --button-primary-border-color-hover: var(--button-primary-border-color);
  --button-primary-text-color: white;
  --button-primary-text-color-hover: var(--button-primary-text-color);
  --button-secondary-background-fill: linear-gradient(to bottom right, var(--neutral-600), var(--neutral-700));
  --button-secondary-background-fill-hover: linear-gradient(to bottom right, var(--neutral-600), var(--neutral-600));
  --button-secondary-border-color: var(--neutral-600);
  --button-secondary-border-color-hover: var(--button-secondary-border-color);
  --button-secondary-text-color: white;
  --button-secondary-text-color-hover: var(--button-secondary-text-color);
  --name: default;
  --primary-50: #fff7ed;
  --primary-100: #ffedd5;
  --primary-200: #fed7aa;
  --primary-300: #fdba74;
  --primary-400: #fb923c;
  --primary-500: #f97316;
  --primary-600: #ea580c;
  --primary-700: #c2410c;
  --primary-800: #9a3412;
  --primary-900: #7c2d12;
  --primary-950: #6c2e12;
  --secondary-50: #eff6ff;
  --secondary-100: #dbeafe;
  --secondary-200: #bfdbfe;
  --secondary-300: #93c5fd;
  --secondary-400: #60a5fa;
  --secondary-500: #3b82f6;
  --secondary-600: #2563eb;
  --secondary-700: #1d4ed8;
  --secondary-800: #1e40af;
  --secondary-900: #1e3a8a;
  --secondary-950: #1d3660;
  --neutral-50: #f9fafb;
  --neutral-100: #f3f4f6;
  --neutral-200: #e5e7eb;
  --neutral-300: #d1d5db;
  --neutral-400: #9ca3af;
  --neutral-500: #6b7280;
  --neutral-600: #4b5563;
  --neutral-700: #374151;
  --neutral-800: #1f2937;
  --neutral-900: #111827;
  --neutral-950: #0b0f19;
  --spacing-xxs: 1px;
  --spacing-xs: 2px;
  --spacing-sm: 4px;
  --spacing-md: 6px;
  --spacing-lg: 8px;
  --spacing-xl: 10px;
  --spacing-xxl: 16px;
  --radius-xxs: 1px;
  --radius-xs: 2px;
  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
  --radius-xl: 12px;
  --radius-xxl: 22px;
  --text-xxs: 9px;
  --text-xs: 10px;
  --text-sm: 12px;
  --text-md: 14px;
  --text-lg: 16px;
  --text-xl: 22px;
  --text-xxl: 26px;
  --font: 'Source Sans Pro', 'ui-sans-serif', 'system-ui', sans-serif;
  --font-mono: 'IBM Plex Mono', 'ui-monospace', 'Consolas', monospace;
  --body-text-size: var(--text-md);
  --body-text-weight: 400;
  --embed-radius: var(--radius-lg);
  --color-accent: var(--primary-500);
  --shadow-drop: rgba(0,0,0,0.05) 0px 1px 2px 0px;
  --shadow-drop-lg: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
  --shadow-inset: rgba(0,0,0,0.05) 0px 2px 4px 0px inset;
  --block-border-width: 1px;
  --block-info-text-size: var(--text-sm);
  --block-info-text-weight: 400;
  --block-label-border-width: 1px;
  --block-label-shadow: var(--block-shadow);
  --block-label-margin: 0;
  --block-label-padding: var(--spacing-sm) var(--spacing-lg);
  --block-label-radius: calc(var(--radius-lg) - 1px) 0 calc(var(--radius-lg) - 1px) 0;
  --block-label-right-radius: 0 calc(var(--radius-lg) - 1px) 0 calc(var(--radius-lg) - 1px);
  --block-label-text-size: var(--text-sm);
  --block-label-text-weight: 400;
  --block-padding: var(--spacing-xl) calc(var(--spacing-xl) + 2px);
  --block-radius: var(--radius-lg);
  --block-shadow: var(--shadow-drop);
  --block-title-background-fill: none;
  --block-title-border-color: none;
  --block-title-border-width: 0px;
  --block-title-padding: 0;
  --block-title-radius: none;
  --block-title-text-size: var(--text-md);
  --block-title-text-weight: 400;
  --container-radius: var(--radius-lg);
  --form-gap-width: 1px;
  --layout-gap: var(--spacing-xxl);
  --panel-border-width: 0;
  --section-header-text-size: var(--text-md);
  --section-header-text-weight: 400;
  --checkbox-border-radius: var(--radius-sm);
  --checkbox-label-gap: var(--spacing-lg);
  --checkbox-label-padding: var(--spacing-md) calc(2 * var(--spacing-md));
  --checkbox-label-shadow: var(--shadow-drop);
  --checkbox-label-text-size: var(--text-md);
  --checkbox-label-text-weight: 400;
  --checkbox-check: url("data:image/svg+xml,%3csvg viewBox='0 0 16 16' fill='white' xmlns='http://www.w3.org/2000/svg'%3e%3cpath d='M12.207 4.793a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0l-2-2a1 1 0 011.414-1.414L6.5 9.086l4.293-4.293a1 1 0 011.414 0z'/%3e%3c/svg%3e");
  --radio-circle: url("data:image/svg+xml,%3csvg viewBox='0 0 16 16' fill='white' xmlns='http://www.w3.org/2000/svg'%3e%3ccircle cx='8' cy='8' r='3'/%3e%3c/svg%3e");
  --checkbox-shadow: var(--input-shadow);
  --error-border-width: 1px;
  --input-border-width: 1px;
  --input-padding: var(--spacing-xl);
  --input-radius: var(--radius-lg);
  --input-shadow: 0 0 0 var(--shadow-spread) transparent, var(--shadow-inset);
  --input-text-size: var(--text-md);
  --input-text-weight: 400;
  --loader-color: var(--color-accent);
  --prose-text-size: var(--text-md);
  --prose-text-weight: 400;
  --prose-header-text-weight: 600;
  --slider-color: #2563eb;
  --table-radius: var(--radius-lg);
  --button-large-padding: var(--spacing-lg) calc(2 * var(--spacing-lg));
  --button-large-radius: var(--radius-lg);
  --button-large-text-size: var(--text-lg);
  --button-large-text-weight: 600;
  --button-shadow: var(--shadow-drop);
  --button-shadow-active: var(--shadow-inset);
  --button-shadow-hover: var(--shadow-drop-lg);
  --button-small-padding: var(--spacing-sm) calc(2 * var(--spacing-sm));
  --button-small-radius: var(--radius-lg);
  --button-small-text-size: var(--text-md);
  --button-small-text-weight: 400;
  --button-transition: none;
}
"""

def predict(img_dict, prompt, strength):
    import time
    start = time.time()
    # 提取涂抹图层
    extract_brush_strokes(img_dict)
    img = img_dict['background']
    img_layer = img_dict['layers'][0]
    
    image_rgb = Image.fromarray(img).convert("RGB")
    image_path = "./data/data.jpg"
    imageio.imwrite(image_path, np.array(image_rgb))

    # 使用box服务提取涂抹处的内容
    x1,y1,x2,y2 = get_bounding_box(img_layer)
    box(x1,y1,x2,y2,'/data2/qinxb/LaMa-demo/data/data.jpg')
    img = load_image(image_path)
    w,h = img.size
    mask_img = load_image('./data/data_mask.jpg')

    # 随机数种子
    generator = torch.manual_seed(33)
    # image = pipe(prompt, image=img,  mask_image=mask_img,  num_inference_steps=25,strength=strength).images[0]  
    image = pipe(prompt, image=img,  generator = generator, mask_image=mask_img,  num_inference_steps=25, strength=strength).images[0]  
    w1,h1 = image.size
    image = image.resize((int(w*h1/h),h1 ))
    image.save("./dataout/result.jpg")
    print(time.time()-start)

    return "./dataout/result.jpg", './data/data_mask.jpg', './data/data_mask_by_box.jpg'

import gradio as gr

if __name__=="__main__":

    with gr.Blocks(css=css) as iface:
        iface.ssl_verify = False
        gr.Markdown("# Knowdee Image Inpainting")

        input_image = gr.ImageEditor(type="numpy",height=1500,width=1000)
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="prompt",value="Fill")
                btn_infer = gr.Button("Run")
            with gr.Column():  
                gr.Markdown("strength越大，添加的噪声越多，与基础图像差异越大，质量越高。")
                strength_num = gr.Slider(minimum=0.2, maximum=1, value=0.9, label="strength")  # 创建滑块
            
        output_inpainted = gr.Image(type="filepath", label="Inpainted Image")

        with gr.Row():
            output_mask = gr.Image(type="filepath", label="Generated Mask")
            output_mask_box = gr.Image(type="filepath", label="Mask Image")

        btn_infer.click(fn=predict, inputs=[input_image, prompt, strength_num], outputs=[output_inpainted, output_mask,output_mask_box])

    iface.launch(server_name="0.0.0.0", server_port=8067, share=False)
