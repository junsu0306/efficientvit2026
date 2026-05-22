import argparse
import os

import onnx
from onnx import helper, TensorProto


def prepend_nhwc_to_nchw(input_path: str, output_path: str):
    model = onnx.load(input_path)
    graph = model.graph

    orig_input = graph.input[0]
    orig_name = orig_input.name
    shape = [d.dim_value for d in orig_input.type.tensor_type.shape.dim]

    if len(shape) != 4:
        raise ValueError(f"Expected 4D input (NCHW), got shape: {shape}")

    N, C, H, W = shape
    nhwc_name = orig_name + "_nhwc"

    nhwc_input = helper.make_tensor_value_info(
        nhwc_name, TensorProto.FLOAT, [N, H, W, C]
    )
    transpose_node = helper.make_node(
        "Transpose",
        inputs=[nhwc_name],
        outputs=[orig_name],
        perm=[0, 3, 1, 2],
    )

    graph.input.remove(orig_input)
    graph.input.insert(0, nhwc_input)
    graph.node.insert(0, transpose_node)

    onnx.checker.check_model(model)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnx.save(model, output_path)
    print(f"[OK] {os.path.basename(input_path)}  ({N},{C},{H},{W}) NCHW  ->  ({N},{H},{W},{C}) NHWC")
    print(f"     saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="입력 ONNX 파일 경로")
    parser.add_argument("--output", type=str, required=True, help="출력 ONNX 파일 경로")
    args = parser.parse_args()

    prepend_nhwc_to_nchw(args.input, args.output)


if __name__ == "__main__":
    main()
