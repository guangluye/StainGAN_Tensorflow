# StainGAN_Tensorflow
A tensorflow version of StainGAN
## How to run 
(1) Run "build_TFRecord.py" to build tfrecords files of the data. Your default input data folders are set as "./data/trainX" and "./data/trainY" and your default output tfrecords files are set as "./data/trainX_tfrecords" and "data/trainY_tfrecords".  
(2) Run "train.py" to train the StainGAN(note that this version only have training process, the testing process process will be added in the future).
## Original Code
[StainGAN](https://github.com/xtarx/StainGAN)