""""""
"""
@authors: Sangamesh Kodge
@copyright: Nanoelectronics Research Laboratory
"""
""""""

import torch.nn as nn

class preprocess(nn.Module):
    """
    This class consists of a forward pass and backward pass function for preprocessing layer. 
    """
    def forward(self, x):
        """
        This function forward propogates through the preprocessing layer
        Args:

        Returns:

        """
        ### Implement the required preprocessing here
        output= x
        return output
    
    def back_approx(self, x):
        """
        This function forward propogates through the preprocessing layer
        Args:
            x : Input to be back propagated
        Returns:
            The backward propagated input 

        """
        ### Implement the required backward function/apprxomation

        output= x
        return output 
                    