import argparse
import onnx

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path of directory saved the input model.')
    parser.add_argument('--input_names', required=True, nargs='+', help='The input name(s) you want to extract.')
    parser.add_argument('--output_names', required=True, nargs='+', help='The output name(s) you want to extract.')
    parser.add_argument('--save_file', required=True, help='Path to save the new onnx model.')
    return parser.parse_args()
        
if __name__ == '__main__':
    args = parse_arguments()
    input_names = []
    output_names = []    
    for name in args.input_names:
        input_names.append(name)
    for name in args.output_names:
        output_names.append(name)    
    onnx.utils.extract_model(args.model, args.save_file, input_names, output_names)
