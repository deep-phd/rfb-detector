#!/usr/bin/env bash
ONNX_PATH='../models/onnx/sim.onnx'
LABEL_PATH='../models/pretrained/voc-model-labels.txt'
SAVE_PATH='./detect_results'
TEST_PATH='./imgs'

python detect_imgs_onnx_without_nms.py \
  --label_path ${LABEL_PATH} \
  --onnx_path ${ONNX_PATH} \
  --test_imgs ${TEST_PATH} \
  --input_size 640 \
  --threshold 0.6 \
  --result_path ${SAVE_PATH} 

