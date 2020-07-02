""""""
"""
@author:  Deepak Ravikumar Tatachar, Sangamesh Kodge
@copyright: Nanoelectronics Research Laboratory
"""
""""""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from  advtrain.utils.normalize import normalize, denormalize
from advtrain.utils.tinyimagenet import TinyImageNet
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

class load_dataset():
    def __init__(self,
                 dataset='CIFAR10',
                 train_batch_size=128, 
                 test_batch_size=128, 
                 val_split=0.0, 
                 augment=True, 
                 padding_crop=4,
                 shuffle=True,
                 random_seed=None,
                 device ='cpu',
                 train_loader=None, 
                 val_loader=None, 
                 test_loader=None, 
                 normalization=lambda x : x,
                 denormalization=lambda x : x,
                 num_classes=None,
                 mean=None, 
                 std=None, 
                 img_ch=None,
                 img_dim=None 
                ):

        '''This is a framework to train and evaluate a network, when training it uses mean and std normalization
        
        Args:
            dataset (str): CIFAR10, CIFAR100, TinyImageNet, ImageNet
            train_batch_size (int): batch size for training dataset
            test_batch_size (int): batch size for testing dataset
            val_split (float): percentage of training data split as validation dataset
            augment (bool): bool flag for Random horizontal flip and shift with padding
            padding_crop (int): units of pixel shift (i.e. used only when augment is True), specifed in the input transformation
            shuffle (bool): bool flag for shuffling the training and validation dataset
            random_seed (int): Fixes the shuffle seed for reproducing the results
            device (str): cuda device or cpu
            train_loader(tuple): dataset loader for mannual data loading of unsupported dataset. 
            val_loader(tuple):dataset loader for mannual data loading of unsupported dataset. Keep val_split=0 for No validation loader. 
            test_loader(tuple):dataset loader for mannual data loading of unsupported dataset
            normalization(function): function normalizing the input data
            denormalization(function): function denormalizing the data required for adversarial attacks.
            num_classes(int): number of classes in the dataset
            mean(float): mean of the normalized data 
            std(float): std of normalised data
            img_ch(int): number of input channels in the input dataset.
            img_dim(int): dimensions(height) of input the input dataset. The class assumes height and widht are equal.
        
        Returns:
            Object of load_dataset 
        '''
        self.dataset = dataset.lower()
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.val_split =val_split
        self.augment = augment
        self.padding_crop = padding_crop
        self.shuffle = shuffle
        self.random_seed = random_seed 
        self.device = device
        if dataset != None:
            dataset_info = self.loader()
        else:
            if train_loader!=None:
                dataset_info ={
                    'train_loader': train_loader, 
                    'validation_loader': val_loader, 
                    'test_loader': test_loader, 
                    'normalization': normalization,
                    'denormalization': denormalization,
                    'num_classes': num_classes, 
                    'mean': mean, 
                    'std': std, 
                    'image_channels': img_ch,
                    'image_dimensions': img_dim 
                }
            else:
                raise ValueError("Unsupported Dataset. Please pass dataset arguments or use the standard supported dataset.")

        self.train_loader = dataset_info['train_loader']
        self.validation_loader = dataset_info['validation_loader']
        self.test_loader = dataset_info['test_loader']
        self.normalization =  dataset_info['normalization']
        self.denormalization = dataset_info['denormalization']
        self.num_classes = dataset_info['num_classes']
        self.mean = dataset_info['mean'] 
        self.std = dataset_info['std']
        self.image_channels = dataset_info['image_channels']
        self.image_dimensions = dataset_info['image_dimensions']
        

    def loader(self, datapath=None):
        '''
        This function returns the dataset parameters for supported datasets
        Args:
            datapath(str) : path to the directory of imagenet/tinyimagenet.

        Returns:
            ``Returns`` a dictionary in the following format::

                {   'train_loader': train_loader, 
                    'validation_loader': val_loader, 
                    'test_loader': test_loader, 
                    'normalization': normalization_function,
                    'denormalization': denormalization_function,
                    'num_classes': num_classes, 
                    'mean': mean, 
                    'std': std, 
                    'image_dimensions': img_dim 
                }
        '''
        # Load dataset
        # Use the following transform for training and testing
        if (self.dataset == 'mnist'):
            mean=[0.1307]
            std=[0.3081]
            img_dim = 28
            img_ch = 1
            num_classes=10
            num_worker = 0
            test_transform = transforms.Compose([
                                                    transforms.ToTensor()
                                                ])
            
            val_transform = test_transform

            if self.augment:
                train_transform = transforms.Compose([
                                                        transforms.RandomCrop(img_dim, padding=self.padding_crop),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                    ])
            else:
                train_transform = test_transform

            trainset = torchvision.datasets.MNIST(root='./data/MNIST/clean',
                                                train=True,
                                                download=True,
                                                transform=train_transform)

            valset = torchvision.datasets.MNIST(root='./data/MNIST/clean',
                                                train=True,
                                                download=True,
                                                transform=val_transform)

            testset = torchvision.datasets.MNIST(root='./data/MNIST/clean',
                                                train=False,
                                                download=True, 
                                                transform=test_transform)     
        elif(self.dataset == 'cifar10'):
            mean=[0.4914, 0.4822, 0.4465]
            std=[0.2023, 0.1994, 0.2010]
            img_dim = 32
            img_ch = 3
            num_classes = 10
            num_worker = 10

            test_transform = transforms.Compose([
                                                    transforms.ToTensor()
                                                ])
            
            val_transform = test_transform

            if self.augment:
                train_transform = transforms.Compose([
                                                        transforms.RandomCrop(img_dim, padding=self.padding_crop),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                    ])
            else:
                train_transform = test_transform

            trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10/',
                                                    train=True,
                                                    download=True,
                                                    transform=train_transform)

            valset = torchvision.datasets.CIFAR10(root='./data/CIFAR10/',
                                                train=True,
                                                download=True,
                                                transform=val_transform)

            testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10/',
                                                train=False,
                                                download=True, 
                                                transform=test_transform)
        elif(self.dataset == 'cifar100'):
            mean = [0.5071, 0.4867, 0.4408]
            std = [0.2675, 0.2565, 0.2761]
            img_dim = 32
            img_ch = 3
            num_classes = 100
            num_worker = 40
            
            test_transform = transforms.Compose([
                                                    transforms.ToTensor()
                                                ])

            val_transform = test_transform

            if self.augment:
                train_transform = transforms.Compose([
                                                transforms.RandomCrop(img_dim, padding=self.padding_crop),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                ])
            else:
                train_transform = test_transform


            trainset = torchvision.datasets.CIFAR100(root='./data/CIFAR100/',
                                                train=True,
                                                download=True,
                                                transform=train_transform)
            valset = torchvision.datasets.CIFAR100(root='./data/CIFAR100/',
                                                train=True,
                                                download=True,
                                                transform=train_transform)
            testset = torchvision.datasets.CIFAR100(root='./data/CIFAR100/',
                                                train=False,
                                                download=True, 
                                                transform=test_transform)    
        elif(self.dataset == 'tinyimagenet'):
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            if datapath==None:
                root = './data/TinyImageNet/'
            else:
                root=datapath
            img_dim = 64
            img_ch = 3
            num_classes = 200
            num_worker = 40
            
            test_transform = transforms.Compose([
                                                    transforms.ToTensor(),
                                                ])
            val_transform = test_transform

            if self.augment:
                train_transform = transforms.Compose([
                                                transforms.RandomResizedCrop(img_dim),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()
                                            ])
            else:
                train_transform= test_transform

            trainset = TinyImageNet(root=root, transform=test_transform, train=True) 
            valset = TinyImageNet(root=root, transform=test_transform, train=True) 
            testset = TinyImageNet(root=root, transform=test_transform, train=False)
        elif(self.dataset == 'imagenet'):
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            if datapath==None:
                datapath = '/local/a/imagenet/imagenet2012/'
                #datapath = '/local/a/wponghir/dataset/imagenet2012/'
            else:
                pass
            img_dim = 224
            img_ch = 3
            num_classes = 1000
            num_worker = 40
                    
            test_transform =  transforms.Compose([
                                                    transforms.Resize(256),
                                                    transforms.CenterCrop(img_dim),
                                                    transforms.ToTensor(),
                                                ])
            val_transform = test_transform
            if self.augment:
                train_transform = transforms.Compose([transforms.Resize(256),
                                                transforms.RandomResizedCrop(img_dim),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()
                                            ])
            else:
                train_transform = test_transform

            trainset = torchvision.datasets.ImageFolder(root=datapath + 'train', transform=train_transform)
            valset =  torchvision.datasets.ImageFolder(root=datapath + 'train', transform=train_transform)
            testset = torchvision.datasets.ImageFolder(root=datapath + 'val', transform=test_transform)
        else:
            # Right way to handle exception in python see https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
            # Explains all the traps of using exception, does a good job!! I mean the link :)
            raise ValueError("Unsupported dataset")
        
        # Split the training dataset into training and validation sets
        print('\nForming the sampler for train and validation split')
        num_train = len(trainset)
        ind = list (range (num_train))
        split = int (np.floor(self.val_split * num_train))

        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(ind)
        
        train_idx, val_idx =ind[split:], ind[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # Load dataloader
        print('Loading data to the dataloader \n')
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch_size, sampler=train_sampler, num_workers=num_worker)
        val_loader =  torch.utils.data.DataLoader(valset, batch_size=self.test_batch_size, sampler=val_sampler, num_workers=num_worker)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.test_batch_size, shuffle=False, num_workers=num_worker)

        _stats = {'mean': mean, 'std': std}
        normalization_params = {**_stats, **{'img_dimensions': [img_ch, img_dim, img_dim]}}
        normalization_function = normalize(**normalization_params, device=self.device)
        denormalization_function = denormalize(**normalization_params, device=self.device)

        return { 'train_loader': train_loader, 
                'validation_loader': val_loader, 
                'test_loader': test_loader, 
                'normalization': normalization_function,
                'denormalization': denormalization_function,
                'num_classes': num_classes, 
                'mean': mean, 
                'std': std, 
                'image_channels': img_ch,
                'image_dimensions': img_dim }