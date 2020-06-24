#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 20:12:54 2019

@author: tibrayev
"""
import torch
import torch.nn as nn
import gc


#def FGSM(clip_min, clip_max, epsilon, image, data_grad):
#    # Collect the element-wise sign of the data gradient
#    sign_data_grad = data_grad.sign()
#    # Create the perturbed image by adjusting each pixel of the input image
#    perturbed_image = image + epsilon * sign_data_grad
#    # Adding clipping to maintain image in the range [clip_min, clip_max]
#    perturbed_image = torch.clamp(perturbed_image, clip_min, clip_max)
#    # Return the perturbed image
#    return perturbed_image

class LinfFGSMSingleStepAttack:
    def __init__(self, clip_min, clip_max, epsilon, loss_func = 'xent'):
        """Attack parameter initialization. The attack performs single step of size epsilon."""
        super(LinfFGSMSingleStepAttack, self).__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon  = epsilon
        if loss_func == 'xent':
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        elif loss_func == 'cw':
            raise ValueError('C&W loss is not implemented!')
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        
    def perturb(self, x_nat, y, model):
        x = x_nat.clone().detach()
        x.requires_grad_(True)
        x.grad = None
        out = model(x)
        loss = self.loss_func(out, y)
        loss.backward()
        grad = x.grad.data # get gradients
        x.requires_grad_(False)
        x.grad = None
        model.zero_grad()
        x += self.epsilon * torch.sign(grad)
        x = torch.clamp(x, self.clip_min, self.clip_max)
        return x

class LinfFGSMSingleStepAttack_with_normalization:
    def __init__(self, clip_min, clip_max, epsilon, loss_func = 'xent'):
        """Attack parameter initialization. The attack performs single step of size epsilon."""
        super(LinfFGSMSingleStepAttack_with_normalization, self).__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon  = epsilon
        if loss_func == 'xent':
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        elif loss_func == 'cw':
            raise ValueError('C&W loss is not implemented!')
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        
    def perturb(self, x_nat, y, model, normalization_function):
        x = x_nat.clone().detach()
        x.requires_grad_(True)
        x.grad = None
        x_norm = normalization_function(x)
        out = model(x_norm)
        loss = self.loss_func(out, y)
        loss.backward()
        grad = x.grad.data # get gradients
        x.requires_grad_(False)
        x.grad = None
        model.zero_grad()
        x += self.epsilon * torch.sign(grad)
        x = torch.clamp(x, self.clip_min, self.clip_max)
        return x

class LinfPGDAttack:
    def __init__(self, clip_min, clip_max, epsilon, k, a, random_start, loss_func = 'xent'):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        super(LinfPGDAttack, self).__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        if loss_func == 'xent':
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        elif loss_func == 'cw':
            raise ValueError('C&W loss is not implemented!')
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        
    def perturb(self, x_nat, y, model):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + (2*self.epsilon)*torch.rand_like(x_nat) - self.epsilon
            x = torch.clamp(x, self.clip_min, self.clip_max)
        else:
            x = x_nat.clone().detach()
        
        for i in range(self.k):
            x.requires_grad_(True)
            x.grad = None
            out = model(x)
            loss = self.loss_func(out, y)
            loss.backward()
            grad = x.grad.data # get gradients
            x.requires_grad_(False)
            x.grad = None
            model.zero_grad()
            x += self.a * torch.sign(grad)
            x = torch.max(torch.min(x, x_nat + self.epsilon), x_nat - self.epsilon)
            x = torch.clamp(x, self.clip_min, self.clip_max)
        return x
    
    def get_grad(self, x, y, model, forward, backward_replacement):
        pre = forward(x)
        preprocessed = pre.clone().detach()
        preprocessed.requires_grad_(True)
        out = model(preprocessed)
        loss = self.loss_func(out, y)
        loss.backward()
        x_clone = x.clone().detach()
        x_clone.requires_grad_(True)
        out = backward_replacement(x_clone)
        torch.autograd.backward(out, grad_tensors=preprocessed.grad.data) 
        return x_clone.grad.data

    def perturb_bpda(self, x_nat, y, model, forward, backward_replacement):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + (2*self.epsilon)*torch.rand_like(x_nat) - self.epsilon
            x = torch.clamp(x, self.clip_min, self.clip_max)
        else:
            x = x_nat.clone().detach()
        
        for i in range(self.k):
            grad = self.get_grad(x, y, model, forward, backward_replacement)
            model.zero_grad()
            x += self.a * torch.sign(grad)
            x = torch.max(torch.min(x, x_nat + self.epsilon), x_nat - self.epsilon)
            x = torch.clamp(x, self.clip_min, self.clip_max)
        return x

class LinfPGDAttack_with_normalization:
    def __init__(self, clip_min, clip_max, epsilon, k, a, random_start, loss_func = 'xent'):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        super(LinfPGDAttack_with_normalization, self).__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        if loss_func == 'xent':
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        elif loss_func == 'cw':
            raise ValueError('C&W loss is not implemented!')
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        
    def perturb(self, x_nat, y, model, normalization_function):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + (2*self.epsilon)*torch.rand_like(x_nat) - self.epsilon
            x = torch.clamp(x, self.clip_min, self.clip_max)
        else:
            x = x_nat.clone().detach()
        
        for i in range(self.k):
            x.requires_grad_(True)
            x.grad = None
            x_norm = normalization_function(x)
            out = model(x_norm)
            loss = self.loss_func(out, y)
            loss.backward()
            grad = x.grad.data # get gradients
            x.requires_grad_(False)
            x.grad = None
            model.zero_grad()
            x += self.a * torch.sign(grad)
            x = torch.max(torch.min(x, x_nat + self.epsilon), x_nat - self.epsilon)
            x = torch.clamp(x, self.clip_min, self.clip_max)
        return x

    def get_grad(self, x, y, model, normalization_function, forward, backward_replacement):
        pre = forward(normalization_function(x))
        preprocessed = pre.clone().detach()
        preprocessed.requires_grad_(True)
        out = model(preprocessed)
        loss = self.loss_func(out, y)
        loss.backward()
        x_clone = x.clone().detach()
        x_clone.requires_grad_(True)
        normalized_x_clone = normalization_function(x_clone)
        out = backward_replacement(normalized_x_clone)
        torch.autograd.backward(out, grad_tensors=preprocessed.grad.data)
        return x_clone.grad.data

    def perturb_bpda(self, x_nat, y, model, normalization_function, forward, backward_replacement, image_at_each_step=False):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        image_list = []
        if self.rand:
            x = x_nat + (2*self.epsilon)*torch.rand_like(x_nat) - self.epsilon
            x = torch.clamp(x, self.clip_min, self.clip_max)
        else:
            x = x_nat.clone().detach()
        
        for i in range(self.k):
            grad = self.get_grad(x, y, model, normalization_function, forward, backward_replacement)
            model.zero_grad()
            x += self.a * torch.sign(grad)
            x = torch.max(torch.min(x, x_nat + self.epsilon), x_nat - self.epsilon)
            x = torch.clamp(x, self.clip_min, self.clip_max)

            if(image_at_each_step):
                x_copy = x.clone().detach()
                image_list.append(x_copy)

        if(image_at_each_step):
            return x, image_list
        return x

    def perturb_bpda_check(self, x_nat, y, model, normalization_function, forward, backward_replacement, repeat=20):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        """ Try multiple times with different start postions around original image
        when we have un succesfull  """
        misclassifed_best = 0
        for i in range(repeat):
            x = self.perturb_bpda(x_nat, y, model, normalization_function, forward, backward_replacement)
            with torch.no_grad():
                output = model(forward(normalization_function(x)))

                # Check for success
                final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                final_pred = final_pred.squeeze(dim=1)
                misclassifed = (final_pred != y).sum()
                if(misclassifed > misclassifed_best):
                    x_best = x.clone().detach()
                
                del x

                if(misclassifed_best == x_nat.shape[0]):
                    break   
        
        gc.collect()
        torch.cuda.empty_cache()
        return x_best

    def perturb_bpda_ensemble(self, x_nat, y, model, normalization_function,attack_type='sign_avg'):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + (2*self.epsilon)*torch.rand_like(x_nat) - self.epsilon
            x = torch.clamp(x, self.clip_min, self.clip_max)
        else:
            x = x_nat.clone().detach()
        
        for i in range(self.k):
            grad = torch.zeros_like(x)
            for quant, net in zip(model.quantization, model.models):
                net.zero_grad()
                forward = quant.forward
                backward_replacement = quant.back_approx
                if(attack_type == 'avg'):
                    grad += self.get_grad(x, y, net, normalization_function, forward, backward_replacement)
                elif(attack_type == 'sign_avg'):
                    grad += torch.sign(self.get_grad(x, y, net, normalization_function, forward, backward_replacement))
                elif(attack_type == 'sign_all'):
                    grad += torch.sign(self.get_grad(x, y, net, normalization_function, forward, backward_replacement))                  
                else:
                    raise ValueError("Unknown attack type specified")
            if (attack_type=='sign_all'):
                x += self.a * torch.where(grad.float().abs()==model.num_models, grad.sign() , torch.zeros_like(grad))
            else:
                x += self.a * torch.sign(grad)
            x = torch.max(torch.min(x, x_nat + self.epsilon), x_nat - self.epsilon)
            x = torch.clamp(x, self.clip_min, self.clip_max)
        
        return x

class LinfRandAttack_with_normalization:
    def __init__(self, clip_min, clip_max, epsilon, k, a):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        super(LinfRandAttack_with_normalization, self).__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon
        self.k = k
        self.a = a

    def get_grad(self, x):
        return torch.rand_like(x)

    def perturb_bpda(self, x_nat, y, model, forward, normalization_function=None, image_at_each_step=False):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        image_list = []
        x = x_nat + (2*self.epsilon)*torch.rand_like(x_nat) - self.epsilon
        x = torch.clamp(x, self.clip_min, self.clip_max)
        
        misclassifed_best = 0
        for j in range(self.k):
            for i in range(self.k):
                grad = self.get_grad(x)
                x += self.a * torch.sign(grad)
                x = torch.max(torch.min(x, x_nat + self.epsilon), x_nat - self.epsilon)
                x = torch.clamp(x, self.clip_min, self.clip_max)

                if(image_at_each_step):
                    x_copy = x.clone().detach()
                    image_list.append(x_copy)

            with torch.no_grad():
                output = model(forward(normalization_function(x)))

                # Check for success
                final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                final_pred = final_pred.squeeze(dim=1)
                misclassifed = (final_pred != y).sum()
                if(misclassifed > misclassifed_best):
                    x_best = x.clone().detach()
        
        gc.collect()
        torch.cuda.empty_cache()

        if(image_at_each_step):
            return x_best, image_list

        return x_best
