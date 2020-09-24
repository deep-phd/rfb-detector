# 轻量级ssd检测模型-RFB Detector

模型以 [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)， 为基础进行训练。标注了2268张训练图片和255张验证图片。

![检测结果](https://github.com/deep-phd/rfb-detector/blob/master/readme_imgs/result_1.jpg"检测结果")

### 训练环境
+ Ubuntu16.04、Ubuntu18.04
+ Python3.6、 Python3.7
+ Pytorch 1.4.0
+ CUDA 10.1 + CUDNN 10.1.243

### 训练
执行命令：`train.sh`

train.sh 内容如下：
```
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
```

参数说明：
- datasets：训练数据路径
- validation_dataset：验证数据路径
- checkpoint_folder：模型保存路径
- net：网络类型
- loss_type：分类损失函数类型
- num_epochs：训练迭代次数
- validation_epochs：训练多少次进行验证及保存模型
- lr：学习率
- debug_steps：训练多少次进行结果打印
- milestones：训练多少次进行学习率调整
- batch_size：batch 大小
- input_size：输入图片尺寸（[n, n, 3]）
- num_workers：处理训练数据的 gpu 个数
- cuda_index：训练用 gpu
***<abbr title="Hyper Text Markup Language">（需要根据具体的需求修改以上参数）</abbr>***


### 推理
- #### 模型转换onnx
进入 test 目录
执行命令：`convert.sh` 即可

convert_to_onnx.sh 内容如下：
```
#!/usr/bin/env bash
MODEL='../models/pretrained/pretrained.pth'
LABEL_PATH='../models/pretrained/voc-model-labels.txt'

python convert_to_onnx.py \
  --label_path ${LABEL_PATH} \
  --model_path ${MODEL} \
  --input_size 640
```

参数说明：
- label_path：label文件的路径
- model_path：待转换模型路径
- input_size：输入图片尺寸（[n, n, 3]）
***<abbr title="Hyper Text Markup Language">（需要根据具体的需求修改以上参数）</abbr>***

- ### 推理
进入 test 目录
执行命令：`test_onnx.sh` 即可

test_onnx.sh 内容如下：
```
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
```

参数说明：
- label_path：label文件的路径
- onnx_path：转换后的 onnx 模型
- test_imgs：待测试用图片
- input_size：输入图片尺寸（[n, n, 3]）
- threshold：检测阈值
- result_path：结果保存路径
***<abbr title="Hyper Text Markup Language">（需要根据具体的需求修改以上参数）</abbr>***

- ### 预训练模型
models/pretrained/pretrained.pth



