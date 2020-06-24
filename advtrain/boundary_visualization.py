""""""
"""
@author: Deepak Ravikumar Tatachar 
@copyright: Nanoelectronics Research Laboratory
"""
""""""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import argparse
import os
import copy
from advtrain.utils.str2bool import str2bool
from advtrain.utils.str2bool import str2bool

class VisualizeBoundaries():
    def __init__(self, 
                 framework=None,
                 net=None,
                 num_classes=10,
                 num_channels=1,
                 img_dim=32,
                 device='cpu'):
        """This is a class to visualize the boundaries of a neural net
        
        Args:
            framework (utils.Framework): Object of Framework, If this is passed none of the other parameters are needed
            net (nn.Module): network whose decision boundaries are desired
            num_classes (int): Number of classes in the dataset
            num_channels (int) : Number of channels in each image of the dataset 
            img_dim (int): Assumes a square image, the height of the image in the dataset
            device (str): cpu/cuda device to perform the evaluation on
        Returns:
            Returns an object of the VisualizeBoundaries
        """   
        if(framework):
            self.net = framework.net
            self.num_classes = framework.dataset_info.num_classes
            self.device = framework.device
            self.num_channels = framework.dataset_info.image_channels
            self.img_size = framework.dataset_info.image_dimensions
        else:
            self.net = net
            self.num_classes = num_classes
            self.device = device
            self.num_channels = num_channels
            self.img_size = img_size

        self.color_lookup = [[1.0, 0.0, 0.0],
                             [1.0, 0.5, 0.0],
                             [1.0, 1.0, 0.0],
                             [0.5, 1.0, 0.0],
                             [0.0, 1.0, 1.0],
                             [0.0, 0.5, 1.0],
                             [0.0, 0.0, 1.0],
                             [0.5, 0.0, 1.0],
                             [1.0, 0.0, 1.0],
                             [0.5, 0.5, 0.5]]

        self.color_lookup = torch.Tensor(self.color_lookup[:self.num_classes]).to(self.device)
        self.num_features = self.img_size * self.img_size * self.num_channels

    def get_color_for_class(self, class_labels):
        '''
        Args:
            class_labels (torch Tensor): Class labels NOT one hot encoded
        Returns:
            returns a 2D tensor with rows as the colors in RGB and the columns are for the corresponding labels
        '''
        one_hot = torch.nn.functional.one_hot(class_labels, num_classes=self.num_classes).float()

        # Convert label to one hot
        pred = torch.stack([one_hot, one_hot, one_hot], dim=2)

        # Multiply the label with the corresponding color values
        color = torch.einsum('ijk, jk-> ijk', pred, self.color_lookup)

        # All except the label index per image will be 0, so sum gives the correct color
        color = color.sum(axis=1)
        return color

    # Return gradient
    def get_gradient(self, x, y):
        criterion = nn.CrossEntropyLoss()
        x_clone = x.clone().detach()
        x_clone.requires_grad_(True)
        out = self.net(x_clone)
        loss = criterion(out, y)
        loss.backward()
        return x_clone.grad.data

    def show(self, image, decision_boundary, index):
        """Visualizes the image and the decision boundary around it
        
        Args:
            image (torch.Tensor): Tensor of images, expected shape [some non zero size, channels, image_dim, image_dim]
            decision_boundary (torch.Tensor): decision boundary a tensor of shape [some non zero size, 3, explore_range, explore_range] which contains an RGB image of the boundaries 
            index (int): A value less than 'some non zero size' and proves an index of the image and decision bounday to display
        Returns:
            Returns a Tensor of shape [batch_size, 3, explore_range, explore_range] which contains an RGB image of the boundaries 
            around the input images.
        """
        plt.subplot(121)
        if self.num_channels==1:
            plt.imshow(image[index][0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        else:    
            plt.imshow(image[index].cpu().numpy().transpose(1,2,0))
        plt.title('Image')
        
        plt.subplot(122)
        plt.imshow(decision_boundary[index].cpu().numpy().transpose(1,2,0))
        plt.title('Decision boundary')
        plt.show()

    def generate_decision_boundaries(self,
                                     images, 
                                     labels,
                                     explore_range=40,
                                     use_random_dir=False):
        """Uses FGSM adversarial/random direction to obtain the boundaries of the nerual net around the specified images

        Args:
            images (torch.Tensor): Tensor of images, expected shape [batch_size, channels, image_dim, image_dim]
            labels (torch.Tensor): Ground truth/correct labels for the images, shape [batch_size,]
            explore_range (int): Specifes the range around the image explored by the code, and empirical value, also determines the size of the output images
            recommended values are, 2, 40, 80, 100. Larger the number more the computations hence it takes longer 
            use_random_dir (bool): Use random direction rather than adversarial direction as the basis vectors for the image 
        
        Returns:
            Returns a Tensor of shape [batch_size, 3, explore_range, explore_range] which contains an RGB image of the boundaries 
            around the input images.
        """
        # Check if image is in format [batch_size, num_channels, img_dim, img_dim]        
        if(len(images.shape) < (self.num_channels + 1)):
            image = image.unsqueeze(0)

        data = images.to(self.device)
        target = labels.to(self.device)
        current_batch_size = data.shape[0]

        if use_random_dir:
            d1 = torch.randn([current_batch_size, self.num_features], device=self.device)
            d1 /= d1.data.norm(dim=1).repeat(self.num_features, 1).transpose(0,1)
        else:
            gradient_vec = self.get_gradient(data, target)
            gradient_l2_norm_value = torch.norm(gradient_vec, p=2, dim=(1,2,3))

            # If norm is 0 replace with 1
            gradient_l2_norm_value = torch.where(gradient_l2_norm_value == 0.00, torch.Tensor([1]).to(self.device), gradient_l2_norm_value)

            expanded_norm = gradient_l2_norm_value.repeat(self.img_size, self.num_channels, self.img_size, 1).transpose(3,0)
            gradient_vec /= expanded_norm

            # Axis 1
            d1 = gradient_vec.view(current_batch_size, -1)

        # Generate perpendicular axis
        # 1) Generate random vector called random_basis
        # 2) Get the component of random_basis along d1
        # 3) Subtract this the componet along d1, from the random vector
        # 4) Normalize
        
        # 1) Generate random vector  
        random_basis = torch.randn([current_batch_size, self.num_features], device=self.device)

        # Calculate component along d1
        # Row wise dot product using Einstien sum
        parallel_component = torch.einsum('ij, ij->i', d1, random_basis)

        # Subtract this the component along d1, from the random vector
        # Axis 2
        d2 = random_basis - torch.einsum('i, ij-> ij', parallel_component, d1)
        d2 /= d2.data.norm(dim=1).repeat(self.num_features, 1).transpose(0,1)

        decision_boundaries = torch.zeros((current_batch_size, 3, 2 * explore_range, 2 * explore_range), device=self.device)

        img_start = data.view(current_batch_size, -1)
        for x in range(-explore_range, explore_range):
            for y in range(-explore_range, explore_range):
                img = torch.clamp(img_start + (20.0 * x / float(explore_range)) * d1 + (20.0 * y / float(explore_range)) * d2 , 0, 1)
                img = img.view(current_batch_size, self.num_channels, self.img_size, self.img_size)
                px = x + explore_range
                py = y + explore_range

                with torch.no_grad():
                    pred = torch.argmax(self.net(img), dim=1)

                decision_boundaries[:,:, px, py] = self.get_color_for_class(pred)
        
        return decision_boundaries