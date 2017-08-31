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
    * 20210 training images (not used)
    * 2000 validation images
* [Portraits 2K](http://xiaoyongshen.me/webpage_portrait/index.html)
    * 1719 training images (used)
* [PASCAL VOC 2012 3K](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)
    * 2913 training images (used)

### Model
* `conv1_1`~`conv5_3`: [VGG-19](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)
* `pool5`~`conv_t3`: Convolution (replace dense network) and Deconvolution

### Directory Structure
```
.Fully-Convolutional-Networks/
├── .Data_zoo/
│   └── .MIT_SceneParsing/
│       ├── ADEChallengeData2016.zip
│       ├── MITSceneParsing.pickle
│       ├── train_data.npz (show up after first train)
│       ├── val_data.npz (show up after first train)
│       └── .ADEChallengeData2016/
│           ├── sceneCategories.txt
│           ├── .images/
│           │   ├── .training/
│           │   └── .validation/
│           │
│           └── .annotations/
│               ├── .training/
│               └── .validation/
│
├── .Model_zoo/
│   └── imagenet-vgg-verydeep-19.mat
│
├── .logs/
│   ├── checkpoint
│   ├── model.ckpt-100000.data-00000-of-00001
│   ├── model.ckpt-100000.meta
│   └── model.ckpt-100000.index
│
├── fcn.py (main program)
├── augment.py
├── batch_datset_reader.py
├── reader.py
├── tensorflow_utils.py
└── README.md

```

* For each data, the filename in `image/`, `annotation/` folder must be same.

### Data Augmentations
* Flip: 50% horizontally
* Rotation: -90 ~ +90
* Scale: 0.5 ~ 1.5
* Shift: -50% ~ +50% horizontally & vertically

### Requirements
* `tensorflow-gpu == 1.2.1`

### Quick Usage
* Train
    * `python3.5 FCN.py -m train`
* Visualize
    * `python3.5 FCN.py -m visualize`
* Test
    * `python3.5 FCN.py -m test -tl <test_list>`
* Mat video
    * `./mat.sh <video_name>`
* To see full usage
    * `python3.5 FCN.py --help`
