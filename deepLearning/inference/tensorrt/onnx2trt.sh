#!/bin/bash

trtexec --onnx=checkpoints/onnx/model_best.onnx --saveEngine=checkpoints/trt/best.trt --workspace=16384 --explicitBatch --device=2

# fp16
trtexec --onnx=checkpoints/onnx/model_best.onnx --saveEngine=checkpoints/trt/bestfp16int8.trt --workspace=16384 --explicitBatch --device=2 --fp16

# dynamic_model
trtexec --onnx=checkpoints/onnx/dynamic_model.onnx --saveEngine=checkpoints/trt/dynamic_model.trt --workspace=16384 --minShapes=input_ids:1x180,token_type_ids:1x180 --optShapes=input_ids:1x180,token_type_ids:1x180 --maxShapes=input_ids:20x180,token_type_ids:20x180 --device=2

# dynamic_model fp16
trtexec --onnx=checkpoints/onnx/dynamic_model.onnx --saveEngine=checkpoints/trt/dynamic_model.trt --workspace=16384 --minShapes=input_ids:1x180,token_type_ids:1x180 --optShapes=input_ids:1x180,token_type_ids:1x180 --maxShapes=input_ids:20x180,token_type_ids:20x180 --device=2 --fp16

ls -ltrh checkpoints/trt/
