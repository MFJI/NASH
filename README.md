# NASH
This repository is the code of Paper: NASH: Neural Architecture Search for Hardware-Optimized Machine Learning Models  
## Training
The code in the resnet18 and resnet34 folders is the code for training the networks.  
Brevitas is needed to train the models. https://github.com/Xilinx/brevitas  
A10 GPU is used in this design.  
The environment of training is as follows:  
  brevitas           0.8.0  
  certifi            2022.12.7  
  charset-normalizer 3.1.0  
  dependencies       2.0.1  
  future-annotations 1.0.0  
  idna               3.4  
  numpy              1.24.2  
  onnx               1.11.0  
  onnxoptimizer      0.2.0  
  packaging          23.1  
  Pillow             9.5.0  
  pip                20.0.2  
  pkg-resources      0.0.0  
  protobuf           3.19.5  
  requests           2.28.2  
  setuptools         44.0.0  
  tokenize-rt        5.0.0  
  torch              1.8.2+cu111  
  torchvision        0.9.2+cu111  
  typing-extensions  4.5.0  
  urllib3            1.26.15  
The environment may needs to be changed based on the GPU you use.
The code train_search.py needs to be run first to train the architecture of the model.  
The next step is to run train_imagenet.py. There is already architecture we have trained for the networks, so it is also possible to run train_imagenet.py without running train_search.py.  
## Pre-trained models
The pre-trained models are in the models folder. These models are already under the architecture we trained.  
## FINN generation
FINN compiler needs to be used to generate the models into hardware. https://github.com/Xilinx/finn  
The code build.py under the finn folder is needed to be run to generate the bit file.  
Alveo FPGA board is the board that needs to be used.  
