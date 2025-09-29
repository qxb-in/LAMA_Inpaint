# LAMA_Inpaint
对图像中的部分区域进行涂抹，将涂抹区域的完整实体识别并提取出，同时使用背景进行图像补全。

在 laptop 端侧实现图像部分抹除与修复，对于 RGBA 图像，取 alpha 通道来判断哪些区
域被涂抹，使用 dreamshaper-8-inpainting 识别涂抹区域中的相关元素边界，并且基于提示词与背景图生成一个新的区域图像。

实例demo如下：

<img width="1627" height="1294" alt="image" src="https://github.com/user-attachments/assets/4d4e04c3-b663-4ac3-bd42-a45a49e4b43e" />

将原图中的小狗涂抹之后抽取出来，并且用新的图像给填充.

# 调整
- model ：dreamshaper-8-inpainting。
- strength ：噪声参数，越大与原图的差异越大，经实测是越大越好，默认为0.9。
- prompt：最好用英文，中文不稳定，实测Fill可能是效果最好的，可以再优化。
- generator：随机数种子，加入generator代表固定随机种子，生成结果固定。
- resize操作：默认的输出结果是一个纵横比1:1的图片，这里做了还原但是较原图还是缩小了，按原图的参数来会变模糊。

Base 开源项目：https://github.com/advimman/lama
