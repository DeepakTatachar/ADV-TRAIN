# ADV-TRAIN : ADV-TRAIN is Deep Vision TRaining And INference framework

This is a framework built on top of pytorch to make machine learning training and inference tasks easier. Along with that it also enables easy dataset and network instantiations, visualize boundaries and more.

Read the latest documentation at https://adv-train.readthedocs.io/en/latest/

## Why use this framework?
- It is very easy to use and well documented and tested
- The framework supports resume (Yes you can restart training from where ever you left off when your server crashed!). 
- The framework also implements support for train/validation splits of your choice with early stopping baked in. 
- Single argument change for using different datasets and models i.e. convenience at you fingertips
- Dataloader parameters optimized for highest possbile performance when traning.
- Supports multi-gpu training (single parameter update required)

### Installing

To install the pip package use the command
```
pip install advtrain
```

### Contributing

To clone the repo, it is recommended to use a shallow clone, this is recommended as previous versions have hosted large pretrained models
```
git clone --depth <specify depth>
```

Requirements are listed in requirements.txt. Use the command

```
pip install -r requirements.txt
```
to install all required dependencies

### Documentation
Read the latest documentation at https://adv-train.readthedocs.io/en/latest/

To locally make the documentation, navigate to /docs and type

```
make html
```

This will generate a build directory and will house a html folder within which you shall find index.html (i.e. path is /docs/build/html/index.html)

Open this in any web browser. This project uses Sphnix to autogenerate this documentation.

### Running Examples

This repo also has examples on how to train and visualize boundaries in /examples folder. A readme file is provided in the ./examples folder to help out with using and running the examples.

### Pretrained Models
We provide pretrained models in a previous version of the repo. It is "hosted" [here](https://github.com/DeepakTatachar/ADV-TRAIN/tree/993c85a80694bcc195526f29f04c1ea80f577437/pretrained). These models have various weight quantized VGG and ResNet models, named according to the naming convention
```
datasetname_inputQuant_architecture_activationQuant_weightQuant.ckpt
```
When running you program using advtrain, place the models in the (current working directory) cwd/pretrained/dataset_name and when load is set to true in instantiate_model it will automatically load the correct model.