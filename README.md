# YOLOv11-TensorRT
use TensorRT to run YOLOv11/YOLOv8, support PTQ quant and dynamic shape

# Introduction

This project provides the code based on python-TensorRT for YOLOv8 or YOLOv11 (they are fully compatible). Note that all onnx here are single-input single-output models.

* PTQ supported
* support pipeline-parallel
* mix-use with pytorch [fix cuda context and stream error]

`notic:` if you run this code and find that it leans towards only `person` detection, don't worry. This is very likely because the main code has been handled, as I am very concerned about the `person` detected results. In fact, the code of YOLO is complete and it fully considers all detect-types, you can fine-tune the external code by yourself.

# usage

```python
from yolo import YOLOv8_11Trt as YOLO
import cv2
yolo = YOLO(
    "model/yolov11m_dynamic.engine",
    confidence_thres=0.5,
    iou_thres=0.5,
    max_batch_size=1,
)

img1=cv2.imread(f"test1.jpg")
img2=cv2.imread(f"test2.jpg")
task1 = np.stack([img1, img2])
detection.detect_sync(task1)                                        # add task1 sync

img3=cv2.imread(f"test3.jpg")
task2 = np.stack([img3])
detection.detect_sync(task2)                                        # add task2 sync

(imgs1, detect_boxs1) = detection.detect_sync_output(wait=True)     # get result for task1, block if computing
img1_result = (imgs1[0], detect_boxs1[0])                           # imgs[i] is raw img, detect_boxs[i] is bbox
img2_result = (imgs1[1], detect_boxs1[1])    

(imgs2, detect_boxs2) = detection.detect_sync_output(wait=True)     # get result for task2
img3_result = (imgs2[0], detect_boxs2[0])

detection.draw_detections(imgs1, detect_boxs1)                      # this function can draw bbox in imgs1. notice: it will modify imgs1 content in-place

yolo.release()
```

# Environment

refer to [how to build environment in ubuntu](env.md)