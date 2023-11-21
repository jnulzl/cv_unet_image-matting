import sys
import onnx
import copy
import onnxsim

onnx_path_1 = sys.argv[1]
onnx_path_2 = sys.argv[2]

onnx_model_1 = onnx.load(onnx_path_1)
onnx_model_2 = onnx.load(onnx_path_2)
io_map = [('simcc_x', 'input1'), ('simcc_y', 'input2')]

onnx_merge = onnx.compose.merge_models(onnx_model_1, onnx_model_2, io_map, 
                                    doc_string="ddddddddddd",
                                    producer_name="jnulzl",
                                    prefix1="rtm_", 
                                    prefix2="head_", 
                                    )

onnx.checker.check_model(onnx_merge)  # check onnx model
onnx_merge, check = onnxsim.simplify(onnx_merge)
onnx.save(onnx_merge, onnx_path_1.replace('.onnx', '_merge.onnx'))