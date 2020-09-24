#!/usr/bin/env bash
MODEL='../models/pretrained/pretrained.pth'
LABEL_PATH='../models/pretrained/voc-model-labels.txt'

python convert_to_onnx.py \
  --label_path ${LABEL_PATH} \
  --model_path ${MODEL} \
  --input_size 640 

