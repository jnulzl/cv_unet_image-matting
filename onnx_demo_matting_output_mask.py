import argparse
import os
import sys
import cv2
import matplotlib.pyplot as plt
import onnxruntime  
import numpy as np  

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path of directory saved the input model.')
    parser.add_argument('--src_img_path', required=True, help='Path to test image.')
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_arguments()
    # 加载 ONNX 模型  
    model_path = args.model
    runtime = onnxruntime.InferenceSession(model_path)  

    # 创建输入数据  
    im_data = cv2.imread(args.src_img_path)
    input_data = im_data.astype(np.float32)
    output_img  = runtime.run(None, {'input_image': input_data})
    output_img = output_img[0].astype(np.uint8)
    cv2.imwrite(args.src_img_path + "_mask.png",output_img)