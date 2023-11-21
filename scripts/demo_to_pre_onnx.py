import os
import sys
import onnx
import torch.onnx
import torch.nn as nn
import onnxsim

def preprocess(nhwc_float_tensor):
    nhwc_float_tensor = nhwc_float_tensor - torch.Tensor([123.675, 116.28 , 103.53])
    nhwc_float_tensor = nhwc_float_tensor * torch.Tensor([0.01712475, 0.017507, 0.01742919])
    nhwc_float_tensor = nhwc_float_tensor.permute(0, 3, 1, 2)
    return nhwc_float_tensor

class PoseIncludePreProcess(nn.Module):
    def __init__(self):
        super(PoseIncludePreProcess, self).__init__()
        pass

    def forward(self, nhwc_float_tensor):
        # ......
        x = preprocess(nhwc_float_tensor)
        return x
                
if __name__ == '__main__':
    
    if 5 != len(sys.argv):
        print("Usage:\n\tpython %s batch_size net_inp_width net_inp_height save_onnx_path"%(sys.argv[0]))
        sys.exit(-1)
    torch_model = PoseIncludePreProcess()    
    batch_size = int(sys.argv[1])
    net_inp_width = int(sys.argv[2])
    net_inp_height = int(sys.argv[3])    
    x = torch.randn(batch_size, net_inp_height, net_inp_width, 3)    
    y = torch_model(x)
    print(y.shape)

    onnx_path = sys.argv[4]
    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      (x),                         # model input (or a tuple for multiple inputs)
                      onnx_path,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=12,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['pre_input'],   # the model's input names
                      output_names = ['pre_output'] # the model's output names
                      # output_names = ['locs', 'max_val_x', 'max_val_y'] # the model's output names
                                    )
                             
    #simplify ONNX...                          
    onnx_model = onnx.load(onnx_path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    onnx_model, check = onnxsim.simplify(onnx_model)
    onnx.save(onnx_model, onnx_path)