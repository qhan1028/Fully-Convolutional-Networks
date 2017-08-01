# Fully Convolutional Networks for Portrait Matting
---
### Basic Knowledges
* [Fully Convolutional Networks](http://simtalk.cn/2016/11/01/Fully-Convolutional-Networks/) (Chinese)

### Referenced Papers
* [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1605.06211v1.pdf)
* [Automatic Portrait Segmentation for Image Stylization](http://xiaoyongshen.me/webpage_portrait/papers/portrait_eg16.pdf)

### Referenced Repositories
* [shekkizh/FCN.tensorflow](https://github.com/shekkizh/FCN.tensorflow)
* [PetroWu/AutoPortraitMatting](https://github.com/PetroWu/AutoPortraitMatting)

### Data Sets
* [MIT Scene Parsing ADE 20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
    * 20210 training images (I picked only 10000)
    * 2000 validation images
* [Portraits 2K](http://xiaoyongshen.me/webpage_portrait/index.html)
    * 1719 training images
* [PASCAL VOC 2012 3K](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)
    * 2913 training images

### Model
* `conv1_1`~`conv5_3`: [VGG-19](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)
* `pool5`~`conv_t3`: Convolution (replace dense network) and Deconvolution

### Directory Structure
```
.Fully-Convolutional-Networks/
├── .Data_zoo/
│   └── ...
├── .Model_zoo/
│   └── imagenet-vgg-verydeep-19.mat
├── .logs/
│   ├── checkpoint
│   ├── model.ckpt-100000.data-00000-of-00001
│   ├── model.ckpt-100000.meta
│   └── model.ckpt-100000.index
├── BatchDatsetReader.py
├── FCN.py
├── read_MITSceneParsingData.py
├── Reader.py
├── TensorflowUtils.py
├── README.md
├── mat.sh
└── train.sh
```

### Requirements
* `tensorflow-gpu == 1.2.1`

### Quick Usage
* Train
    * `python3.5 FCN.py -m train`
      or
    * `./train.sh`
* Visualize
    * `python3.5 FCN.py -m visualize`
* Test
    * `python3.5 FCN.py -m test -tl <test_list>`
* Mat video
    * `./mat.sh <video_name>`
* To see full usage
    * `python3.5 FCN.py --help`
