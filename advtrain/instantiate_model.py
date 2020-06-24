""""""
"""
@author:  Deepak Ravikumar Tatachar, Sangamesh Kodge
@copyright: Nanoelectronics Research Laboratory
"""
""""""
import os
import torch
from advtrain.models.AlexNet import *
from advtrain.models.resnet import *
from advtrain.models.lenet5 import *
from advtrain.models.vgg import *
import torchvision.models as models
from advtrain.utils.quantise import Quantise2d
from advtrain.utils.preprocess import preprocess as PreProcess

def instantiate_model (dataset='cifar10',
                       num_classes=10, 
                       input_quant='FP', 
                       arch='resnet',
                       dorefa=False, 
                       abit=32, 
                       wbit=32,
                       qin=False, 
                       qout=False,
                       suffix='', 
                       load=False,
                       torch_weights=False,
                       device='cpu'):
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
    #Select the input transformation
    if input_quant==None:
        input_quant=''
        Q=PreProcess()
    elif input_quant.lower()=='q1':
        Q = Quantise2d(n_bits=1).to(device)
    elif input_quant.lower()=='q2':
        Q = Quantise2d(n_bits=2).to(device)
    elif input_quant.lower()=='q4':
        Q = Quantise2d(n_bits=4).to(device)
    elif input_quant.lower()=='q6':
        Q = Quantise2d(n_bits=6).to(device)
    elif input_quant.lower()=='q8':
        Q = Quantise2d(n_bits=8).to(device)
    elif input_quant.lower()=='fp':
        Q = Quantise2d(n_bits=1,quantise=False).to(device)
    else:    
        raise ValueError

    # Instantiate model1
    # RESNET IMAGENET
    if(arch == 'torch_resnet18'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.resnet18(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_resnet34'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.resnet34(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_resnet50'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.resnet50(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_resnet101'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.resnet101(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_resnet152'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.resnet152(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_resnet34'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.resnet34(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_resnext50_32x4d'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.resnext50_32x4d(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_resnext101_32x8d'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.resnext101_32x8d(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_wide_resnet50_2'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.wide_resnet50_2(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_wide_resnet101_2'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.wide_resnet101_2(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    #VGG IMAGENET
    elif(arch == 'torch_vgg11'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.vgg11(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_vgg11bn'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.vgg11_bn(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_vgg13'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.vgg13(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_vgg13bn'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.vgg13_bn(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_vgg16'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.vgg16(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_vgg16bn'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.vgg16_bn(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_vgg19'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.vgg19(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_vgg19bn'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.vgg19_bn(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    #MOBILENET IMAGENET   
    elif(arch == 'torch_mobnet'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.mobilenet_v2(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    #DENSENET IMAGENET
    elif(arch == 'torch_densenet121'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.densenet121(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_densenet169'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.densenet169(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_densenet201'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.densenet201(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif(arch == 'torch_densenet161'):
        if dorefa:
            raise ValueError ("Dorefa net unsupported for {}".format(arch))
        else:
            model = models.densenet161(pretrained=torch_weights)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    #RESNET CIFAR   
    elif(arch == 'resnet' or arch == 'resnet18'  ):
        if dorefa:
            model = ResNet18_Dorefa(num_classes=num_classes, abit=abit, wbit=wbit)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch +"_a" + str(abit) + 'w'+ str(wbit) + suffix
        else:
            model = ResNet18(num_classes=num_classes)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif( arch == 'resnet34'  ):
        if dorefa:
            model = ResNet34_Dorefa(num_classes=num_classes, abit=abit, wbit=wbit)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch +"_a" + str(abit) + 'w'+ str(wbit) + suffix
        else:
            model = ResNet34(num_classes=num_classes)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif( arch == 'resnet50'  ):
        if dorefa:
            model = ResNet50_Dorefa(num_classes=num_classes, abit=abit, wbit=wbit)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch +"_a" + str(abit) + 'w'+ str(wbit) + suffix
        else:
            model = ResNet50(num_classes=num_classes)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif( arch == 'resnet101'  ):
        if dorefa:
            model = ResNet101_Dorefa(num_classes=num_classes, abit=abit, wbit=wbit)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch +"_a" + str(abit) + 'w'+ str(wbit) + suffix
        else:
            model = ResNet101(num_classes=num_classes)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    elif( arch == 'resnet152'  ):
        if dorefa:
            model = ResNet152_Dorefa(num_classes=num_classes, abit=abit, wbit=wbit)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch +"_a" + str(abit) + 'w'+ str(wbit) + suffix
        else:
            model = ResNet152(num_classes=num_classes)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
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
        if dorefa:
            model = vgg_Dorefa(cfg=cfg, batch_norm_conv=batch_norm_conv, batch_norm_linear=batch_norm_linear ,num_classes=num_classes, a_bit=abit, w_bit=wbit)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch +"_a" + str(abit) + 'w'+ str(wbit) + suffix
            
        else:   
            model = vgg(cfg=cfg, batch_norm_conv=batch_norm_conv, batch_norm_linear=batch_norm_linear ,num_classes=num_classes)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    # LENET MNIST
    elif (arch == 'lenet5'):
        if dorefa:
            model = LeNet5_Dorefa(num_classes=num_classes, abit=abit, wbit=wbit)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch +"_a" + str(abit) + 'w'+ str(wbit) + suffix
        else:
            model = LeNet5(num_classes=num_classes)
            model_name = dataset.lower()+ "_" + input_quant + "_" + arch + suffix
    else:
        # Right way to handle exception in python see https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
        # Explains all the traps of using exception, does a good job!! I mean the link :)
        raise ValueError("Unsupported neural net architecture")
    model = model.to(device)
    
    if load == True and torch_weights == False :
        print(" Using Model: " + arch)
        model_path = os.path.join('./pretrained/', dataset.lower(),  model_name + '.ckpt')
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        print(' Loaded trained model from :' + model_path)
        print(' {}'.format(Q))
    
    else:
        model_path = os.path.join('./pretrained/', dataset.lower(),  model_name + '.ckpt')
        print(' Training model save at:' + model_path)
    print('')
    return model, model_name, Q