## BSHM人像抠图ONNX版本
*原始模型见：*[modelscope-cv_unet_image-matting](https://www.modelscope.cn/models/damo/cv_unet_image-matting/summary)

### 安装依赖

```
>>python --version
Python 3.9.13
>>pip install -r requirements.txt
```

### demo

- 1、直接输出抠完的图像
```
python onnx_demo_matting_output_png.py --model sim/cv_unet_image_matting_opset11_output_png.onnx --src_img_path imgs/1.png
```

**原图：**

![wx](D:\codes\Python\BSHM\cv_unet_image_matting\imgs\wx.jpg)

**人像扣图：**

![wx.jpgmerge](D:\codes\Python\BSHM\cv_unet_image_matting\imgs\wx.jpgmerge.png)

- 2、只输出人像`mask`
```
python onnx_demo_matting_output_mask.py --model sim/cv_unet_image_matting_opset11_output_float_mask.onnx --src_img_path imgs/1.png
```

**原图：**

![wx](D:\codes\Python\BSHM\cv_unet_image_matting\imgs\wx.jpg)

**人像`mask`：**

![wx.jpg_mask](D:\codes\Python\BSHM\cv_unet_image_matting\imgs\wx.jpg_mask.png)
