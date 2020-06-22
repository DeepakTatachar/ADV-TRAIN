"""
@author:  Deepak Ravikumar Tatachar, Sangamesh Kodge
@copyright: Nanoelectronics Research Laboratory
"""
import torch
from models.AlexNet import *
from models.resnet import *
from models.lenet5 import *
from models.vgg import *
import torchvision.models as models
import os

def instantiate_model(dataset='cifar10',
                      num_classes=10, 
                      arch='resnet',
                      suffix='', 
                      load=False,
                      torch_weights=False):

    """Initializes/load network with random weight/saved and return auto generated model name 'dataset_arch_suffix.ckpt'
    
    Args:
        dataset         : mnists/cifar10/cifar100/imagenet/tinyimagenet/simple dataset the netwoek is trained on. Used in model name 
        num_classes     : number of classes in dataset. 
        arch            : resnet/vgg/lenet5/basicnet/slpconv model architecture the network to be instantiated with 
        suffix          : str appended to the model name 
        load            : boolean variable to indicate load pretrained model from ./pretrained/dataset/
        torch_weights   : boolean variable to indicate load weight from torchvision for imagenet dataset
    Returns:
        model           : models with desired weight (pretrained / random )
        model_name      : str 'dataset_arch_suffix.ckpt' used to save/load model in ./pretrained/dataset
    """

    # RESNET IMAGENET
    if(arch == 'torch_resnet18'):
        model = models.resnet18(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_resnet34'):
        model = models.resnet34(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_resnet50'):
        model = models.resnet50(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_resnet101'):
        model = models.resnet101(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_resnet152'):
        model = models.resnet152(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_resnet34'):
        model = models.resnet34(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_resnext50_32x4d'):
        model = models.resnext50_32x4d(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_resnext101_32x8d'):
        model = models.resnext101_32x8d(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_wide_resnet50_2'):
        model = models.wide_resnet50_2(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_wide_resnet101_2'):
        model = models.wide_resnet101_2(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix

    #VGG IMAGENET
    elif(arch == 'torch_vgg11'):
        model = models.vgg11(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_vgg11bn'):
        model = models.vgg11_bn(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_vgg13'):
        model = models.vgg13(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_vgg13bn'):
        model = models.vgg13_bn(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_vgg16'):
        model = models.vgg16(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_vgg16bn'):
        model = models.vgg16_bn(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_vgg19'):
        model = models.vgg19(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_vgg19bn'):
        model = models.vgg19_bn(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix

    #MOBILENET IMAGENET   
    elif(arch == 'torch_mobnet'):
        model = models.mobilenet_v2(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix

    #DENSENET IMAGENET
    elif(arch == 'torch_densenet121'):
        model = models.densenet121(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_densenet169'):
        model = models.densenet169(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_densenet201'):
        model = models.densenet201(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif(arch == 'torch_densenet161'):
        model = models.densenet161(pretrained=torch_weights)
        model_name = dataset.lower()+ "_" + arch + suffix

    #RESNET CIFAR   
    elif(arch == 'resnet' or arch == 'resnet18'  ):
        model = ResNet18(num_classes=num_classes)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif( arch == 'resnet34'  ):
        model = ResNet34(num_classes=num_classes)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif( arch == 'resnet50'  ):
        model = ResNet50(num_classes=num_classes)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif( arch == 'resnet101'  ):
        model = ResNet101(num_classes=num_classes)
        model_name = dataset.lower()+ "_" + arch + suffix
    elif( arch == 'resnet152'  ):
        model = ResNet152(num_classes=num_classes)
        model_name = dataset.lower()+ "_" + arch + suffix

    #VGG CIFAR
    elif(arch[0:3] == 'vgg'):
        len_arch = len(arch)
        if arch[len_arch-2:len_arch]=='bn' and arch[len_arch-4:len_arch-2]=='bn':
            batch_norm_conv=True
            batch_norm_linear=True
            cfg= arch[3:len_arch-4]
        elif arch [len_arch-2: len_arch]=='bn':
            batch_norm_conv=True
            batch_norm_linear=False
            cfg= arch[3:len_arch-2]
        else:
            batch_norm_conv=False
            batch_norm_linear=False
            cfg= arch[3:len_arch]
        model = vgg(cfg=cfg, batch_norm_conv=batch_norm_conv, batch_norm_linear=batch_norm_linear ,num_classes=num_classes)
        model_name = dataset.lower()+ "_" + arch + suffix

    # LENET MNIST
    elif (arch == 'lenet5'):
        model = LeNet5(num_classes=num_classes)
        model_name = dataset.lower()+ "_" + arch + suffix
    else:
        # Right way to handle exception in python see https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
        # Explains all the traps of using exception, does a good job!! I mean the link :)
        raise ValueError("Unsupported neural net architecture")
    
    if load == True and torch_weights == False :
        print(" Using Model: " + arch)
        model_path = os.path.join('./pretrained/', dataset.lower(),  model_name + '.ckpt')
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        print(' Loaded trained model from :' + model_path)
    
    else:
        model_path = os.path.join('./pretrained/', dataset.lower(),  model_name + '.ckpt')
        print(' Training model save at:' + model_path)
    print('')
    return model, model_name