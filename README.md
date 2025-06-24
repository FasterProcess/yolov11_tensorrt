# YOLOv11-TensorRT
use TensorRT to run YOLOv11/YOLOv8, support PTQ quant and dynamic shape

# Introduction

This project provides the code based on python-TensorRT for YOLOv8 or YOLOv11 (they are fully compatible). Note that all onnx here are single-input single-output models.

* PTQ supported
* support pipeline-parallel
* mix-use with pytorch [fix cuda context and stream error]

`notic:` if you run this code and find that it leans towards only `person` detection, don't worry. This is very likely because the main code has been handled, as I am very concerned about the `person` detected results. In fact, the code of YOLO is complete and it fully considers all detect-types, you can fine-tune the external code by yourself.

# env

```bash
# apt install nvidia-driver-535

# install CUDA-12.4 + cuDNN 9.10.2
## omitted how to install CUDA-12.4, here
cat >> ~/.bashrc <<EOF
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
# nvcc -V to check CUDA version

wget https://developer.download.nvidia.com/compute/cudnn/9.10.2/local_installers/cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb
dpkg -i cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb
cp /var/cudnn-local-repo-ubuntu2204-9.10.2/cudnn-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cudnn-cuda-12

# install TensorRT
###################################################### TensorRT Backend ##########################################################
# CPP version need envs below
# export TENSORRT_HOME=/usr/local/TensorRT/TensorRT-10.4.0.26
# export LD_LIBRARY_PATH=$PATH:$TENSORRT_HOME/lib:$LD_LIBRARY_PATH
# export LIBRARY_PATH=$PATH:$TENSORRT_HOME/lib:$LIBRARY_PATH
# export C_INCLUDE_PATH=$C_INCLUDE_PATH:$TENSORRT_HOME/include
# export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$TENSORRT_HOME/include
###############################################################################################################################
pip3 install tensorrt==10.4
```

# Addition

* [YOLOv11 with ONNXRuntime](https://github.com/oneflyingfish/yolov11-onnxruntime)
* [YOLOv11 with TensorRT](https://github.com/oneflyingfish/yolov11_tensorrt)
* [DeepSort+YOLO11 with TensorRT](https://github.com/oneflyingfish/yolo_deepsort_tensorrt)