""""""
"""
@authors: Timur Ibrayev, Deepak Ravikumar Tatachar
@copyright: Nanoelectronics Research Laboratory
"""
""""""
import torch

class normalize(object):
    '''
    Normalizes the input and provides a backpropable normalization function. It perfroms the channel wise normalization
    using mean and standard deviation

    Args:
        mean (list): list of the mean for each channel
        std (list): list of the std for each channel
        img_dimensions (list): list [channels,h,w]
        device (str): cpu/cuda

    Returns:
        ``Returns`` an object of denormalize
    '''
    def __init__(self, mean, std, img_dimensions, device='cpu'):
        self.mean = mean
        self.std  = std
        self.img_dims = img_dimensions
        self.device = device
        self.channels = img_dimensions[0]
        self.mean_expanded, self.std_expanded = self.get_mean_std_expanded_tensors()
        
    def get_mean_std_expanded_tensors(self):
        channel_expanded_mean_tensor_list = []
        channel_expanded_std_tensor_list = []
        for channel in range(self.channels):
            channel_expanded_mean = torch.ones((1, self.img_dims[1], self.img_dims[2])) * self.mean[channel]
            channel_expanded_std = torch.ones((1, self.img_dims[1], self.img_dims[2])) * self.std[channel]

            channel_expanded_mean_tensor_list.append(channel_expanded_mean)
            channel_expanded_std_tensor_list.append(channel_expanded_std)

        mean_expanded = torch.cat(channel_expanded_mean_tensor_list, dim=0).to(self.device)
        std_expanded = torch.cat(channel_expanded_std_tensor_list, dim=0).to(self.device)
        return mean_expanded, std_expanded

    def __call__(self, tensor):
        assert tensor.ndimension() == 4, 'Input image is not 4D!'
        normalized_tensor = tensor.sub(self.mean_expanded.expand_as(tensor)).div(self.std_expanded.expand_as(tensor))
        return normalized_tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, imgae dim={2}, channels={3})'.format(self.mean, self.std, self.img_dims, self.channels)

class denormalize(object):
    '''
    De-normalizes the input and provides a backpropable de-normalization function
    
    Args:
        mean (list): list of the mean for each channel
        std (list): list of the std for each channel
        img_dimensions (list): list [channels,h,w]
        device (str): cpu/cuda

    Returns:
        ``Returns`` an object of normalize
    '''
    def __init__(self, mean, std, img_dimensions, device='cpu'):
        self.mean = mean
        self.std  = std
        self.img_dims = img_dimensions
        self.device = device
        self.channels = img_dimensions[0]
        self.mean_expanded, self.std_expanded = self.get_mean_std_expanded_tensors()

    def get_mean_std_expanded_tensors(self):
        channel_expanded_mean_tensor_list = []
        channel_expanded_std_tensor_list = []
        for channel in range(self.channels):
            channel_expanded_mean = torch.ones((1, self.img_dims[1], self.img_dims[2])) * self.mean[channel]
            channel_expanded_std = torch.ones((1, self.img_dims[1], self.img_dims[2])) * self.std[channel]

            channel_expanded_mean_tensor_list.append(channel_expanded_mean)
            channel_expanded_std_tensor_list.append(channel_expanded_std)

        mean_expanded = torch.cat(channel_expanded_mean_tensor_list, dim=0).to(self.device)
        std_expanded = torch.cat(channel_expanded_std_tensor_list, dim=0).to(self.device)
        return mean_expanded, std_expanded

    def __call__(self, tensor):
        assert tensor.ndimension() == 4, 'Input image is not 4D!'
        normalized_tensor = tensor.mul_(self.mean_expanded.expand_as(tensor)).add(self.std_expanded.expand_as(tensor))
        return normalized_tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, imgae dim={2}, channels={3})'.format(self.mean, self.std, self.img_dims, self.channels)