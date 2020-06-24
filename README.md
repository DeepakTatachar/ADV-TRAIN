# ADV-TRAIN : ADV-TRAIN is Deep Vision TRaining And INference framework

This is a framework built on top of pytorch to make machine learning training and inference tasks easier. Along with that it also enables easy dataset and network instantiations, visualize boundaries and more.

Read the latest documentation at https://adv-train.readthedocs.io/en/latest/

## Why use this framework?
- It is very easy to use and well documented and tested
- The framework supports resume (Yes you can restart training from where ever you left off when your server crashed!). 
- The framework also implements support for train/validation splits of your choice with early stopping baked in. 
- Single argument change for using different datasets and models i.e. convinence at you fingertips
- Dataloader parameters optimized for highest possbile performance when traning.
- Supports multi-gpu training (single parameter update required)

### Installing

Requirements are listed in requirements.txt. Use the command

```
pip install -r requirements.txt
```
to install all required dependencies

### Documentation
Read the latest documentation at https://adv-train.readthedocs.io/en/latest/

To read the documentation, navigate to /docs and type

```
make html
```

This will generate a build directory and will house a html folder within which you shall find index.html (i.e. path is /docs/build/html/index.html)

Open this in any web browser. This project uses Sphnix to autogenerate this documentation.

### Running Examples

This repo has examples on how to train and visualize boundaries in /examples folder.
When using the training code please create the following folder structure in the root directory (this is autommatically created)

```
/pretrained/<dataset name in small letters>/temp
```

This lets the framework store the models with the nomenclature datasetname_architecture_suffix.ckpt. The temp folder contains information stored by the framework for resume support.

### Pretrained Models
We provide pretrained models in ./pretrained folder