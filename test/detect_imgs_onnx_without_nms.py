"""
This code uses the onnx model to detect faces from live video or cameras.
"""
import os
import time
import random

import cv2
import numpy as np
import onnx
import sys
import pickle as pkl

sys.path.append("..")
from caffe2.python.onnx import backend
import vision.utils.box_utils_numpy as box_utils
from vision.ssd.config import fd_config
from vision.ssd.config.fd_config import define_img_size
import torch

# onnx runtime
import onnxruntime as ort

#input_img_size = 320
input_img_size = 640
define_img_size(input_img_size)

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

label_path = "../models/objdetect/voc-model-labels.txt"
onnx_path = "../models/onnx/sim.onnx"
class_names = [name.strip() for name in open(label_path).readlines()]

predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)
onnx.helper.printable_graph(predictor.graph)
predictor = backend.prepare(predictor, device="CPU")  # default CPU

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name
result_path = "./detect_results"

threshold = 0.8
path = "./imgs"
sum = 0
if not os.path.exists(result_path):
    os.makedirs(result_path)
listdir = os.listdir(path)
sum = 0
colors = pkl.load(open("pallete", "rb"))
for file_path in listdir:
    img_path = os.path.join(path, file_path)
    orig_image = cv2.imread(img_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image, (320, 240))
    image = cv2.resize(image, (640, 480))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    # confidences, boxes = predictor.run(image)
    time_time = time.time()
    confidences, boxes = ort_session.run(None, {input_name: image})

    ############
    boxes = box_utils.convert_locations_to_boxes(
       boxes, fd_config.priors, fd_config.center_variance, fd_config.size_variance
       #torch.from_numpy(boxes), fd_config.priors, fd_config.center_variance, fd_config.size_variance
    )
    boxes = box_utils.center_form_to_corner_form(boxes)
    ############

    print("cost time:{}".format(time.time() - time_time))
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)

    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"

        color = random.choice(colors)
        c1 = (box[0], box[1])
        c2 = (box[2], box[3])
        cv2.rectangle(orig_image, c1, c2,color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 220, c1[1] + t_size[1] + 25
        cv2.rectangle(orig_image, c1, c2,color, -1)
        cv2.putText(orig_image, label, (c1[0], c1[1] + t_size[1] + 24), cv2.FONT_HERSHEY_PLAIN, 3, [225,255,255], 2);

        cv2.imwrite(os.path.join(result_path, file_path), orig_image)

    sum += boxes.shape[0]

print("sum:{}".format(sum))

