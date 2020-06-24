
"""
author: Sangamesh Kodge

Modified from github: https://github.com/zzzxxxttt/pytorch_DoReFaNet/blob/master/utils/quant_dorefa.py

@copyright: Nanoelectronics Research Laboratory
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def uniform_quantize(k):
  """This functions returns a class of uniform quantization

  Args: 
    k : bit precision of quantization 

  Returns:
    A class with uniform quantization with backward and forward functions.
  """
  class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      # elif k == 1:
      #   out = torch.sign(input)
      else:
        n = float(2 ** k - 1)
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply


class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit):
    """This class quantization weights
    
    Args:
      a_bit(int)  : bit precision for weight quantization

    Returns:
      Class object for weight quantization 
    """
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 16 or w_bit == 32
    self.w_bit = w_bit
    self.uniform_q = uniform_quantize(k=w_bit)

  def forward(self, x):
    """This is a function that quantizes the weights 

        Args:
            x(tensor)    : weights data to be quantized

        Returns:
            Returns quantized tensor
        """   
    if self.w_bit == 32:
      weight_q = x
    # elif self.w_bit == 1:
    #   E = torch.mean(torch.abs(x)).detach()
    #   weight_q = self.uniform_q(x / E) * E
    else:
      weight = torch.tanh(x)
      max_w = torch.max(torch.abs(weight)).detach()
      weight = weight / 2 / max_w + 0.5
      weight_q = max_w * (2 * self.uniform_q(weight) - 1)
    return weight_q


class activation_quantize_fn(nn.Module):
  
  def __init__(self, a_bit):
    """This class quantization activations
    
    Args:
      a_bit(int)  : bit precision for activation quantization

    Returns:
      Class object for activation quantization 
    """
  
    super(activation_quantize_fn, self).__init__()
    assert a_bit <= 16 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = uniform_quantize(k=a_bit)

  def forward(self, x):
    """This is a function that quantizes the activation 

        Args:
            x(tensor)    : activation data to be quantized

        Returns:
            Returns quantized tensor
        """   
        
    if self.a_bit == 32:
      activation_q = x
    else:
      activation_q = self.uniform_q(torch.clamp(x, 0, 1))
      # print(np.unique(activation_q.detach().numpy()))
    return activation_q


def conv2d_Q_fn(w_bit):
  """This is a function returns a conv2d Layer class which has quantized weights

    Args:
      w_bits (int)  : bit precision for weight quantization

    Returns:
      A class for conv2d with weight quantization. Parameters of the class same as torch.nn.Conv2d
        """    
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input, order=None):  
        
      weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.conv2d(input, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

  return Conv2d_Q


def linear_Q_fn(w_bit):
  """This is a function returns a Linear Layer class which has quantized weights

    Args:
      w_bits (int)  : bit precision for weight quantization

    Returns:
      A class for fully connected layer with weight quantization.Parameters of the class same as torch.nn.Linear
        """   
  class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
      super(Linear_Q, self).__init__(in_features, out_features, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input):
      weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.linear(input, weight_q, self.bias)

  return Linear_Q


if __name__ == '__main__':
  import numpy as np
  import matplotlib.pyplot as plt

  a = torch.rand(1, 3, 32, 32)

  Conv2d = conv2d_Q_fn(w_bit=1)
  conv = Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

  img = torch.randn(1, 256, 56, 56)
  print(img.max().item(), img.min().item())
  out = conv(img)
  print(out.max().item(), out.min().item())