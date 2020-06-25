# ADV-TRAIN : Examples
This directory has two main examples, more will be added. This document explains how to use the example programs

## train.py
train.py is your one stop solution to training any image recognition model on popular dataset. The framework aloows you to quantize the inputs, weights activations using dorefa net (see https://arxiv.org/abs/1606.06160v1).

To train a full precision, resnet50, model on cifar10, for 10 epochs use:
```
python train.py --arch=resnet50 --dataset=cifar10 --input_quant=FP --epochs=10
```

Similarly you can train a vgg19, on cifar100 with 4 bit weight quantization with 4 bit input quantization for 15 epochs by using
```
python train.py --arch=vgg19 --dataset=cifar100 --input_quant=Q4 --wbit=Q4 --epochs=15
```

The train.py is a very versatile tool for DL-Researchers

## visualize_boundaries.py
This is an example, which trains the model and then visualizes the decision boundaries around the inputs. (Kindly refer to Section 6 of the paper https://arxiv.org/pdf/1611.02770.pdf)