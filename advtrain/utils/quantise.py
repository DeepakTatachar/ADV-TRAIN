"""
@author:  Sangamesh Kodge

@copyright: Nanoelectronics Research Laboratory
"""

import torch
import torch.nn as nn

class Quantise2d(nn.Module):

    def __init__(self, n_bits=1, quantise=True, un_normalized=False):
        """This is a class that quantizes the input 
        
        Args:
            n_bits (int)        : bit precision of input quantization
            quantise (bool)     : True for quantization false for passing the input as is
            un_normalized(bool) : True if input is un_normalized i.e between [0, 255], false otherwise.
        Returns:
            Returns an object of the Quantise2d
        """   
        
        super(Quantise2d, self).__init__()
        assert n_bits <= 8.0
        self.n_bits = n_bits
        self.quantise=quantise
        self.un_normalized=un_normalized
        
    
    def forward(self, data):
        """This is a function that quantizes the input 

        Args:
            data(tensor)    : input data to be quantized

        Returns:
            Returns quantized tensor
        """   
        if self.quantise:
            if self.un_normalized and self.n_bits<8:

                data = (data * 255.0).long().float()
                bin_width = 255.0/2**(self.n_bits)
                data=torch.where(data==255.0,data-0.5 * bin_width * torch.ones_like(data), data)
                data = torch.floor( data/bin_width ) * bin_width + 0.5 * bin_width * torch.ones_like(data)
                data = data / 255.0

            elif(not(self.un_normalized)) :

                bin_width = (data.max()-data.min())/ 2**self.n_bits
                data=torch.where(data==data.max(),data-0.5*bin_width,data)
                data = torch.floor( (data - torch.ones_like(data)*data.min() )/ bin_width)  * bin_width  + torch.ones_like(data) * (0.5 * bin_width + data.min())

            else:

                raise ValueError            
        return data
    
    def back_approx(self, x):
        """This is a function does a linear backward pass approximation for quantization
        
        Args:
            data(tensor)    : input data for backprop
            
        Returns:
            Returns backprop tensor tensor
        """ 
        
        return x

    def __repr__(self):
        if(self.quantise):
            return "Quantize2d, bits:" + str(self.n_bits)
        else:
            return "Quantize2d FP"

    def __str__(self):
        if(self.quantise):
            return "Quantize2d, bits:" + str(self.n_bits)
        else:
            return "Quantize2d FP"
    


    