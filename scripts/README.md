## 一些有用的onnx模型处理脚本

- 1、dynamic_shape to fixed_shape


```shell
# input shape:[batch,3,192,192] -> [2,3,192,192]
python make_onnx_dynamic_shape_to_fixed.py  --onnx_path files/test_origin.onnx --input_name input --input_shape 2,3,192,192
```

**抓换前的模型头部shape**

![test_origin_input](D:\codes\Python\BSHM\cv_unet_image_matting\scripts\files\test_origin_input.png)

**抓换后的模型头部shape**

![test_origin_bs2_input](D:\codes\Python\BSHM\cv_unet_image_matting\scripts\files\test_origin_bs2_input.png)

- 2、rename onnx model node name


```shell
# input name:'input' -> 'inp'
# output name:'simcc_x' -> 'output_x' && 'simcc_y' -> 'output_y'
python rename_onnx_model.py --model files/test_origin.onnx --origin_names 'input' 'simcc_x' 'simcc_y'  --new_names 'inp' 'output_x' 'output_y'  --save_file files/test_origin_renamed.onnx
```

**改名前的模型头部**

![test_origin_before_rename](D:\codes\Python\BSHM\cv_unet_image_matting\scripts\files\test_origin_before_rename.png)

**改名后的模型头部**

![test_origin_renamed](D:\codes\Python\BSHM\cv_unet_image_matting\scripts\files\test_origin_renamed.png)

- 3、extract sub onnx model


```shell
# sub onnx input name:'input'
# sub onnx output name:'simcc_x'
python extract_sub_onnx.py --model files/test_origin.onnx --input_names 'input' --output_names 'simcc_x' --save_file files/test_origin_sub.onnx
```

**原始模型头部**

![test_origin_before_rename](D:\codes\Python\BSHM\cv_unet_image_matting\scripts\files\test_origin_before_rename.png)

**子模型头部**

![test_origin_renamed](D:\codes\Python\BSHM\cv_unet_image_matting\scripts\files\test_origin_sub.png)

- 4、merge two onnx to one 

利用`demo_to_pre_onnx.py`生成预处理模型，
利用`onnx_merge.py`将预处理模型与另一个onnx模型进行合并

参考[这里](https://onnx.ai/onnx/api/compose.html#merge-models)

