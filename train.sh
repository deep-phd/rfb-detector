#!/usr/bin/env bash
model_root_path="./models/objdetect"
TRAINING_DATA='./training_data'

python train.py \
  --datasets ${TRAINING_DATA} \
  --validation_dataset ${TRAINING_DATA} \
  --checkpoint_folder ${model_root_path} \
  --net RFB \
  --loss_type focal \
  --num_epochs 500 \
  --validation_epochs 10 \
  --lr 1e-2 \
  --debug_steps 10 \
  --milestones "150" \
  --batch_size 64 \
  --input_size 640 \
  --num_workers 4 \
  --cuda_index 4,5,6,7 \
  2>&1 | tee "$log"

