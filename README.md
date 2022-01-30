#### (NOTICE) All the ReXNet-lite's model files have been updated!
#### (NOTICE) Our paper has been accepted at CVPR 2021!! The paper has been updated at [arxiv](https://arxiv.org/pdf/2007.00992.pdf)!

## Rethinking Channel Dimensions for Efficient Model Design

**Dongyoon Han, Sangdoo Yun, Byeongho Heo, and YoungJoon Yoo** | [Paper](https://arxiv.org/abs/2007.00992) | [Pretrained Models](#pretrained)

NAVER AI Lab

## Abstract

Designing an efficient model within the limited computational cost is challenging. We argue the accuracy of a lightweight model has been further limited by the design convention: a stage-wise configuration of the channel dimensions, which looks like a piecewise linear function of the network stage. In this paper, we study an effective channel dimension configuration towards better performance than the convention. To this end, we empirically study how to design a single layer properly by analyzing the rank of the output feature. We then investigate the channel configuration of a model by searching network architectures concerning the channel configuration under the computational cost restriction. Based on the investigation, we propose a simple yet effective channel configuration that can be parameterized by the layer index. As a result, our proposed model following the channel parameterization achieves remarkable performance on ImageNet classification and transfer learning tasks including COCO object detection, COCO instance segmentation, and fine-grained classifications. 

## Model performance
- We first illustrate our models' top-acc. vs. computational costs graphs compared with EfficientNets


<img src=https://user-images.githubusercontent.com/31481676/113254746-f0416500-9301-11eb-9cd8-f188037cc82c.png width=2000 hspace=20>


### Performance comparison
#### ReXNets vs EfficientNets
- The CPU latencies are tested on Xeon E5-2630_v4 with a single image and the GPU latencies are measured on a V100 GPU with **the batchsize of 64**.
- EfficientNets' scores are taken form [arxiv v3 of the paper](https://arxiv.org/pdf/1905.11946v3.pdf).

    Model | Input Res. | Top-1 acc. | Top-5 acc. | FLOPs/params. | CPU Lat./ GPU Lat.
     :--: |:--:|:--:|:--:|:--:|:--:|
    **ReXNet_0.9** | 224x224 | 77.2 | 93.5 | 0.35B/4.1M | 45ms/20ms
    |||||    
    EfficientNet-B0 | 224x224 | 77.3 | 93.5 |  0.39B/5.3M | 47ms/23ms  
    **ReXNet_1.0** | 224x224 | 77.9 | 93.9 | 0.40B/4.8M | 47ms/21ms
    |||||
    EfficientNet-B1 | 240x240 | 79.2 | 94.5 | 0.70B/7.8M | 70ms/37ms
    **ReXNet_1.3** | 224x224 | 79.5 | 94.7| 0.66B/7.6M | 55ms/28ms  
    |||||
    EfficientNet-B2 | 260x260 | 80.3 | 95.0 | 1.0B/9.2M | 77ms/48ms
    **ReXNet_1.5** | 224x224 | 80.3 | 95.2| 0.88B/9.7M | 59ms/31ms
    |||||
    EfficientNet-B3 | 300x300 | 81.7 | 95.6 | 1.8B/12M | 100ms/78ms    
    **ReXNet_2.0** | 224x224 | 81.6 | 95.7 |  1.8B/19M | 69ms/40ms 
    
#### ReXNet-lites vs. EfficientNet-lites
- ReXNet-lites do not use SE-net an SiLU activations aiming to faster training and inference speed.
- We compare ReXNet-lites with [EfficientNet-lites](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite).
- Here the GPU latencies are measured on two M40 GPUs, we will update the number run on a V100 GPU soon.

  Model | Input Res. | Top-1 acc. | Top-5 acc. | FLOPs/params | CPU Lat./ GPU Lat.
  :--: |:--:|:--:|:--:|:--:|:--:|
  EfficientNet-lite0 | 224x224 | 75.1 | - |  0.41B/4.7M | 30ms/49ms
  **ReXNet-lite_1.0** | 224x224 | 76.2 | 92.8 | 0.41B/4.7M | 31ms/49ms
  |||||
  EfficientNet-lite1 | 240x240 | 76.7 | - |  0.63B/5.4M | 44ms/73ms
  **ReXNet-lite_1.3** | 224x224 | 77.8 | 93.8 | 0.65B/6.8M | 36ms/61ms  
  |||||  
  EfficientNet-lite2 | 260x260 | 77.6 | - | 0.90B/ 6.1M | 48ms/93ms
  **ReXNet-lite_1.5** | 224x224 | 78.6 | 94.2| 0.84B/8.3M| 39ms/68ms    
  |||||  
  EfficientNet-lite3 | 280x280| 79.8  | - |  1.4B/ 8.2M | 60ms/131ms
  **ReXNet-lite_2.0** | 224x224 | 80.2  | 95.0 | 1.5B/13M | 49ms/90ms  
  
## ImageNet-1k Pretrained models
<h2 id="pretrained"> ImageNet classification results</h2>

- Please refer the following pretrained models. Top-1 and top-5 accuraies are reported with the computational costs.
- Note that all the models are trained and evaluated with 224x224 image size.

  Model | Input Res. | Top-1 acc. | Top-5 acc. | FLOPs/params | 
  :--: |:--:|:--:|:--:|:--:
  [ReXNet_1.0](https://drive.google.com/file/d/1xeIJ3wb83uOowU008ykYj6wDX2dsncA9/view?usp=sharing)  | 224x224 | 77.9 | 93.9 | 0.40B/4.8M | 
  [ReXNet_1.3](https://drive.google.com/file/d/1x2ziK9Oyv66Y9NsxJxXsdjzpQF2uSJj0/view?usp=sharing)  | 224x224 | 79.5 | 94.7 | 0.66B/7.6M | 
  [ReXNet_1.5](https://drive.google.com/file/d/1TOBGsbDhTHWBgqcRnyKIR0tHsJTOPUIG/view?usp=sharing)  | 224x224 | 80.3 | 95.2 | 0.88B/9.7M | 
  [ReXNet_2.0](https://drive.google.com/file/d/1R1aOTKIe1Mvck86NanqcjWnlR8DY-Z4C/view?usp=sharing)  | 224x224 | 81.6 | 95.7 | 1.5B/16M | 
  [ReXNet_3.0](https://drive.google.com/file/d/1iXAsr8gs3pRz0QyHKomdj5SGVzPWbIs2/view?usp=sharing)  | 224x224 | 82.8 | 96.2 | 3.4B/34M |  
  ||||
  [ReXNet-lite_1.0](https://drive.google.com/file/d/1d9G4pLwZwkoDR2TRPCQlxiWiuC7R-Oqf/view?usp=sharing) | 224x224 | 76.2 | 92.8 | 0.41B/4.7M |
  [ReXNet-lite_1.3](https://drive.google.com/file/d/1NsbsdI8qAHG6HdMxmySXcrl9NdEx3s0L/view?usp=sharing) | 224x224 | 77.8 | 93.8 | 0.65B/6.8M |
  [ReXNet-lite_1.5](https://drive.google.com/file/d/12QzIh9A-U0PBGaLNOIr4gX2MoZEBnRjk/view?usp=sharing) | 224x224 | 78.6 | 94.2 | 0.84B/8.3M| 
  [ReXNet-lite_2.0](https://drive.google.com/file/d/1pGdG9HWnqSAu1FajmaMJMK5JyOJaiFyW/view?usp=sharing) | 224x224 | 80.2 | 95.0 | 1.5B/13M | 

### Finetuning results
#### COCO Object detection 
- The following results are trained with **Faster RCNN with FPN**:

  | Backbone |Img. Size|  B_AP (%) | B_AP_0.5 (%) |  B_AP_0.75 (%) | Params. |FLOPs | Eval. set|
  |:----:|:----:|:----:|:----:|:----:|:---:|:---:|:---:|
  | FBNet-C-FPN        | 1200x800 | 35.1 | 57.4 | 37.2 | 21.4M | 119.0B | val2017 |
  | EfficientNetB0-FPN | 1200x800 | 38.0 | 60.1 | 40.4 | 21.0M | 123.0B | val2017|
  | ReXNet_0.9-FPN     | 1200x800 | 38.0 | **60.6** | 40.8 | 20.1M | 123.0B | val2017|
  | ReXNet_1.0-FPN     | 1200x800 | **38.5** | **60.6** | **41.5** | 20.7M | 124.1B | val2017|
  |||||||||
  | ResNet50-FPN     | 1200x800 | 37.6| 58.2| 40.9 | 41.8M | 202.2B | val2017|
  | ResNeXt-101-FPN  | 1200x800 | 40.3 | 62.1 | 44.1 | 60.4M | 272.4B | val2017|
  | ReXNet_2.2-FPN   | 1200x800| **41.5** | **64.0** | **44.9** | 33.0M | 153.8B | val2017|


#### COCO instance segmentation
- The following results are trained with **Mask RCNN with FPN**, S_AP and B_AP denote segmentation AP and box AP, respectively:

  | Backbone |Img. Size|  S_AP (%) | S_AP_0.5 (%) | S_AP_0.75 (%) | B_AP (%) | B_AP_0.5 (%) | B_AP_0.75 (%) | Params. |FLOPs | Eval. set|
  |:----:|:----:|:----:|:----:|:----:|:---:|:---:|:---:|:---:|:---:|:---:|
  | EfficientNetB0_FPN     | 1200x800 | 34.8 | 56.8 | 36.6 | 38.4 | 60.2 | 40.8 | 23.7M | 123.0B | val2017|
  | ReXNet_0.9-FPN  | 1200x800 | **35.2** | **57.4**| **37.1** |**38.7** |**60.8**|**41.6**| 22.8M | 123.0B | val2017|
  | ReXNet_1.0-FPN  | 1200x800 | 35.4 | 57.7 | 37.4 | 38.9 |61.1 | 42.1 | 23.3M | 124.1B | val2017|
  |||||||||||| 
  | ResNet50-FPN           | 1200x800 | 34.6 | 55.9 | 36.8 |38.5 |59.0|41.6|  44.2M | 207B | val2017|
  | ReXNet_2.2-FPN | 1200x800 | **37.8** | **61.0** | **40.2** | **42.0** | **64.5** | **45.6**|  35.6M | 153.8B | val2017|
  
## Getting Started
### Requirements
- Python3
- PyTorch (> 1.0)
- Torchvision (> 0.2)
- NumPy

### Using the pretrained models
- [timm>=0.3.0](https://github.com/rwightman/pytorch-image-models) provides the wonderful wrap-up of ours models thanks to [Ross Wightman](https://github.com/rwightman). Otherwise, the models can be loaded as follows:
  - To use ReXNet on a GPU:
  ```python
  import torch
  import rexnetv1
  
  model = rexnetv1.ReXNetV1(width_mult=1.0).cuda()
  model.load_state_dict(torch.load('./rexnetv1_1.0.pth'))
  model.eval()
  print(model(torch.randn(1, 3, 224, 224).cuda()))
  ```

  - To use ReXNet-lite on a CPU:
  ```python
  import torch
  import rexnetv1_lite

  model = rexnetv1_lite.ReXNetV1_lite(multiplier=1.0)
  model.load_state_dict(torch.load('./rexnet_lite_1.0.pth', map_location=torch.device('cpu')))
  model.eval()
  print(model(torch.randn(1, 3, 224, 224)))

  ```

### Training own ReXNet

ReXNet can be trained with any PyTorch training codes including [ImageNet training in PyTorch](https://github.com/pytorch/examples/tree/master/imagenet) with the model file and proper arguments. Since the provided model file is not complicated, we simply convert the model to train a ReXNet in other frameworks like MXNet. For MXNet, we recommend [MXnet-gluoncv](https://gluon-cv.mxnet.io/model_zoo/classification.html) as a training code.

Using PyTorch, we trained ReXNets with one of the popular imagenet classification code, [Ross Wightman](https://github.com/rwightman)'s [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) for more efficient training. After including ReXNet's model file into the training code, one can train ReXNet-1.0x with the following command line:

    ./distributed_train.sh 4 /imagenet/ --model rexnetv1 --rex-width-mult 1.0 --opt sgd --amp \
     --lr 0.5 --weight-decay 1e-5 \
     --batch-size 128 --epochs 400 --sched cosine \
     --remode pixel --reprob 0.2 --drop 0.2 --aa rand-m9-mstd0.5 
     
Using droppath or MixUP may need to train a bigger model.

## License

This project is distributed under [MIT license](LICENSE).


## How to cite

```
@misc{han2021rethinking,
      title={Rethinking Channel Dimensions for Efficient Model Design}, 
      author={Dongyoon Han and Sangdoo Yun and Byeongho Heo and YoungJoon Yoo},
      year={2021},
      eprint={2007.00992},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
