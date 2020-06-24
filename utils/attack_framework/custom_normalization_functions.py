#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:01:51 2019

@author: tibrayev
"""
import torch
class custom_1channel_img_normalization_with_dataset_params(object):
    def __init__ (self, mean, std, img_dimensions, device='cpu'):
        self.mean = mean
        assert len(self.mean) == 1, 'Custom norm function is for 1 channel images. Expected 1 elements for mean, got {}'.format(len(mean))
        self.std  = std
        assert len(self.std) == 1, 'Custom norm function is for 1 channel images. Expected 1 elements for std, got {}'.format(len(std))
        self.img_dims = img_dimensions
        assert len(self.img_dims) == 3, 'Custom norm function is for 1 channel images. Expected 3 elements for img_dimensions, got {}'.format(len(img_dimensions))
        assert self.img_dims[0] == 1, 'Custom norm function is for 1 channel images. Expected 1 channels in img_dimensions, got {}'.format(img_dimensions[0])
        self.in_device = device
        
        self.mean_expanded = (torch.ones((1, self.img_dims[1], self.img_dims[2]))*self.mean[0]).to(self.in_device)
        self.std_expanded = (torch.ones((1, self.img_dims[1], self.img_dims[2]))*self.std[0]).to(self.in_device)
        
    def __call__(self, tensor):
        assert tensor.ndimension() == 4, 'Input image is not 4D!'
        normalized_tensor = tensor.sub(self.mean_expanded.expand_as(tensor)).div(self.std_expanded.expand_as(tensor))
        return normalized_tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, img_dims={2})'.format(self.mean, self.std, self.img_dims)
    
class custom_1channel_img_UnNormalize_with_dataset_params(object):
    def __init__(self, mean, std,img_dimensions, device='cpu'):
        self.mean = mean
        self.std = std
        self.img_dims = img_dimensions
        
        assert len(self.mean) == 1, 'Custom norm function is for 1 channel images. Expected 1 elements for mean, got {}'.format(len(mean))
        self.std  = std
        assert len(self.std) == 1, 'Custom norm function is for 1 channel images. Expected 1 elements for std, got {}'.format(len(std))
        self.img_dims = img_dimensions
        assert len(self.img_dims) == 3, 'Custom norm function is for 1 channel images. Expected 3 elements for img_dimensions, got {}'.format(len(img_dimensions))
        assert self.img_dims[0] == 1, 'Custom norm function is for 1 channel images. Expected 1 channels in img_dimensions, got {}'.format(img_dimensions[0])
        self.in_device = device
        self.mean_expanded = (torch.ones((1, self.img_dims[1], self.img_dims[2]))*self.mean[0]).to(self.in_device)
        
        self.std_expanded = (torch.ones((1, self.img_dims[1], self.img_dims[2]))*self.std[0]).to(self.in_device)

    def __call__(self, tensor):
        assert tensor.ndimension() == 4, 'Input image is not 4D!'
        tensor = tensor.mul_(self.mean_expanded.expand_as(tensor)).add(self.std_expanded.expand_as(tensor))
        return tensor
    

class custom_3channel_img_normalization_with_dataset_params(object):
    def __init__ (self, mean, std, img_dimensions, device='cpu'):
        self.mean = mean
        assert len(self.mean) == 3, 'Custom norm function is for 3 channel images. Expected 3 elements for mean, got {}'.format(len(mean))
        self.std  = std
        assert len(self.std) == 3, 'Custom norm function is for 3 channel images. Expected 3 elements for std, got {}'.format(len(std))
        self.img_dims = img_dimensions
        assert len(self.img_dims) == 3, 'Custom norm function is for 3 channel images. Expected 3 elements for img_dimensions, got {}'.format(len(img_dimensions))
        assert self.img_dims[0] == 3, 'Custom norm function is for 3 channel images. Expected 3 channels in img_dimensions, got {}'.format(img_dimensions[0])
        self.in_device = device
        
        self.mean_expanded = torch.cat((torch.ones((1, self.img_dims[1], self.img_dims[2]))*self.mean[0],
                                        torch.ones((1, self.img_dims[1], self.img_dims[2]))*self.mean[1],
                                        torch.ones((1, self.img_dims[1], self.img_dims[2]))*self.mean[2]
                                        ), dim = 0).to(self.in_device)
        
        self.std_expanded = torch.cat((torch.ones((1, self.img_dims[1], self.img_dims[2]))*self.std[0],
                                       torch.ones((1, self.img_dims[1], self.img_dims[2]))*self.std[1],
                                       torch.ones((1, self.img_dims[1], self.img_dims[2]))*self.std[2]
                                       ), dim = 0).to(self.in_device)
        
    def __call__(self, tensor):
        assert tensor.ndimension() == 4, 'Input image is not 4D!'
        normalized_tensor = tensor.sub(self.mean_expanded.expand_as(tensor)).div(self.std_expanded.expand_as(tensor))
        return normalized_tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, img_dims={2})'.format(self.mean, self.std, self.img_dims)


class custom_3channel_img_UnNormalize_with_dataset_params(object):
    def __init__(self, mean, std,img_dimensions, device='cpu'):
        self.mean = mean
        self.std = std
        self.img_dims = img_dimensions
        
        assert len(self.mean) == 3, 'Custom norm function is for 3 channel images. Expected 3 elements for mean, got {}'.format(len(mean))
        self.std  = std
        assert len(self.std) == 3, 'Custom norm function is for 3 channel images. Expected 3 elements for std, got {}'.format(len(std))
        self.img_dims = img_dimensions
        assert len(self.img_dims) == 3, 'Custom norm function is for 3 channel images. Expected 3 elements for img_dimensions, got {}'.format(len(img_dimensions))
        assert self.img_dims[0] == 3, 'Custom norm function is for 3 channel images. Expected 3 channels in img_dimensions, got {}'.format(img_dimensions[0])
        self.in_device = device
        self.mean_expanded = torch.cat((torch.ones((1, self.img_dims[1], self.img_dims[2]))*self.mean[0],
                                        torch.ones((1, self.img_dims[1], self.img_dims[2]))*self.mean[1],
                                        torch.ones((1, self.img_dims[1], self.img_dims[2]))*self.mean[2]
                                        ), dim = 0).to(self.in_device)
        
        self.std_expanded = torch.cat((torch.ones((1, self.img_dims[1], self.img_dims[2]))*self.std[0],
                                        torch.ones((1, self.img_dims[1], self.img_dims[2]))*self.std[1],
                                        torch.ones((1, self.img_dims[1], self.img_dims[2]))*self.std[2]
                                        ), dim = 0).to(self.in_device)

    def __call__(self, tensor):
        assert tensor.ndimension() == 4, 'Input image is not 4D!'
        tensor = tensor.mul_(self.mean_expanded.expand_as(tensor)).add(self.std_expanded.expand_as(tensor))
        return tensor
    


class custom_3channel_img_normalization_with_per_image_params(object):
    def __init__ (self, img_dimensions):
        self.img_dims = img_dimensions
        assert len(self.img_dims) == 3, 'Custom norm function is for 3 channel images. Expected 3 elements for img_dimensions, got {}'.format(len(img_dimensions))
        assert self.img_dims[0] == 3, 'Custom norm function is for 3 channel images. Expected 3 channels in img_dimensions, got {}'.format(img_dimensions[0])
        self.img_dims_flat = self.img_dims[0] * self.img_dims[1] * self.img_dims[2]
        self.minstd = torch.tensor([1.0/torch.sqrt(torch.tensor([self.img_dims_flat*1.0]))])
        
    def __call__(self, tensor):
        assert tensor.ndimension() == 4, 'Input image is not 4D!'
        # flat each image pixels to [batch_size, num_of_img_pixels]
        imgs_flat = tensor.view(tensor.size(0), -1)
        # compute mean for each image over all img pixels (regardless of image channels)
        per_img_mean = imgs_flat.mean(dim=1, keepdim=True)
        # compute std for each image over all img pixels (regardless of image channels)
        per_img_std  = imgs_flat.std(dim=1, keepdim=True)
        # in case the image has uniform distribution, adjust std to minimum standard deviation
        per_img_std_adjusted = torch.max(per_img_std, self.minstd)
        
        # normalize input tensor
        normalized_tensor_flat = imgs_flat.sub(per_img_mean).div(per_img_std_adjusted)
        
        return normalized_tensor_flat.view_as(tensor)
    
    def __repr__(self):
        return self.__class__.__name__ + '(img_dims={0})'.format(self.img_dims)