# NASH
This repository is the code of Paper: NASH: Neural Architecture Search for Hardware-Optimized Machine Learning Models  
## Training
The code in the resnet18 and resnet34 folders is the code for training the networks.  
Brevitas is needed to train the models. https://github.com/Xilinx/brevitas  
The environment of training is as follows:  
  
The code train_search.py needs to be run first to train the architecture of the model.  
The next step is to run train_imagenet.py. There is already architecture we have trained for the networks, so it is also possible to run train_imagenet.py without running train_search.py.  
## Pre-trained models
The pre-trained models are in the models folder. These models are already under the architecture we trained.  
## FINN generation
FINN compiler needs to be used to generate the models into hardware. https://github.com/Xilinx/finn
The code build.py under the finn folder is needed to be run.  
