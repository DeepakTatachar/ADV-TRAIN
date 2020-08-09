""""""
"""
@authors: Sangamesh Kodge
@copyright: Nanoelectronics Research Laboratory
"""
""""""
import torch
from advtrain.load_dataset import load_dataset
import matplotlib.pyplot as plt
from advtrain.attack_framework.multi_lib_attacks import attack_wrapper
from advtrain.utils.preprocess import preprocess as PreProcess
from advtrain.framework import Framework
import os


class Blackbox_extention():
    def __init__(self,
                 framework,
                 net=None,
                 model_name=None,
                 dataset='cifar10',
                 normalize=None,
                 preprocess=None,
                 epochs=350,
                 optimizer='sgd',
                 loss='crossentropy',
                 learning_rate=0.01,
                 device=None):
        """This is a framework to train and evaluate a network, when training it uses mean and std normalization
        
        Args:
            net (nn.Module): network to be trained
            dataset (str): CIFAR10, CIFAR100, TinyImageNet, ImageNet
            preprocess (object): a callable object, lambda or nn.Module whose forward implements the preprocessing step
            train_batch_size (int): batch size for training dataset
            test_batch_size (int): batch size for testing dataset
            val_split (float): percentage of training data split as validation dataset
            augment (bool): bool flag for Random horizontal flip and shift with padding
            padding_crop (int): units of pixel shift (i.e. used only when augment is True), specifed in the input transformation
            shuffle (bool): bool flag for shuffling the training and validation dataset
            random_seed (int): Fixes the shuffle seed for reproducing the results
            optimizer (str): Name of the optimizer to use
            loss (str): Name of the loss to use for training
            adversarial_training (bool): True for adversarial training
            lib (str): select the implementing library custom, advertorch or foolbox
            attack(str): select the attack type PGD,CW,...
            iterations(int): Number of iterations for the attack
            epsilon(float) : attack strength
            stepsize(float) : step size for the attack
            use_bpda : Backward propagation through differential approximation
            target(int): None for non targeted attack. class label for targeted attack
            random(bool) : False for deterministic implementation
            device (str): cpu/cuda device to perform the evaluation on, if left to None picks up the CPU if available
        Returns:
            Returns an object of the framework
        """   
        self.net = net
        self.dataset = framework.dataset.lower()
        self.normalize = normalize
        self.train_batch_size = framework.train_batch_size
        self.test_batch_size = framework.test_batch_size
        self.val_split = framework.val_split
        self.framework = framework
        self.num_epochs = epochs
        self.model_name = model_name
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        
        if(device is None):
            self.device = framework.device
        else:
            self.device = device

        # Optimizer
        self.optimizer = self.get_optimizer(optimizer, learning_rate)
        if self.val_split>0:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        else:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=range(0,500,100),gamma=0.1)
        
        self.criterion = self.get_criterion_for_loss_function(loss)
        self.dataset_info = framework.dataset_info

        self.train_loader = self.dataset_info.train_loader
        self.val_loader = self.dataset_info.validation_loader
        self.test_loader = self.dataset_info.test_loader

        self.train_len = len(self.train_loader)
        self.val_len = len(self.val_loader)
        self.test_len = len(self.test_loader)

        if(self.normalize is None):
            self.normalize = self.dataset_info.normalization
        self.num_channels = self.dataset_info.image_channels

        self.num_classes = self.dataset_info.num_classes
        self.img_size = self.dataset_info.image_dimensions

        if preprocess==None:
            self.preprocess =PreProcess()
        else:
            self.preprocess =preprocess
        

        os.makedirs('./pretrained/', exist_ok=True)
        os.makedirs('./pretrained/'+self.dataset+'/', exist_ok=True)
        os.makedirs('./pretrained/'+self.dataset+'/temp/', exist_ok=True)

        if self.framework.adversarial_training:
            self.attack_info = self.framework.attack_info
            self.attack = attack_wrapper(self.net, self.device, **self.attack_info)
            self.targeted = self.framework.targeted
            self.target = self.framework.target

        
        # Training parameters exposed for hooks to access them
        self.best_val_accuracy = 0.0
        self.best_val_loss = 0.0
        self.current_train_loss = 0.0
        self.current_train_acc = 0.0
        self.current_epoch = 0
        self.current_batch = 0
        self.saves = []
        self.current_batch_data = {}

        # Saves the current test accuracy for the parameters obtained or the best validation accuracy
        self.current_test_acc = 0.0

    def update_network(self, new_net, new_model_name):
        '''
        Updates all the internal states to make sure new model can be trained correctly 
        Args:
            new_net (nn.Module): New netwrok to train
            new_model_name (str): name of new model to save
        
        Returns:
            No return data 
        '''
        self.net = new_net
        self.model_name = new_model_name
        self.optimizer = self.get_optimizer(self.optimizer_name, self.learning_rate)

        self.best_val_accuracy = 0.0
        self.best_val_loss = 0.0
        self.current_train_loss = 0.0
        self.current_train_acc = 0.0
        self.current_epoch = 0
        self.current_batch = 0
        self.saves = []
        self.current_batch_data = {}

    def get_criterion_for_loss_function(self, loss):
        '''
        Args:
            loss (str): name of loss
        
        Returns:
            Returns the loss_criteria from torch 
        '''
        loss = loss.lower()
        if loss == 'crossentropy' or loss == 'xentropy':
            return torch.nn.CrossEntropyLoss()
            
        elif loss == 'mse':
            return torch.nn.MSELoss()

        else:
            raise ValueError ("Unsupported loss function")

    def get_optimizer(self, optimizer, learning_rate):
        '''
        Args:
            optimizer (str): name of optimizer
            learning_rate (float): learning rate of optimizer
        
        Returns:
            Returns the corresponding torch optimizer
        '''
        optimizer = optimizer.lower()
        if optimizer =='sgd':
            return torch.optim.SGD(self.net.parameters(),
                                   lr=learning_rate,
                                   momentum=0.9,
                                   weight_decay=5e-4)

        elif optimizer =='adagrad':
            return torch.optim.Adagrad(self.net.parameters(), 
                                       lr=learning_rate)

        elif optimizer =='adam':
            return torch.optim.Adam(self.net.parameters(),
                                    lr=learning_rate)

        else:
            raise ValueError ("Unsupported Optimizer")

    def adversarial_attack(self):
        """Adversarial Inference function

        Returns:
            correct, total, accuracy.
            correct (int), the number of correctly classifed images
            total (int), the total number of images in the test set
            accuracy (float), accuracy in %
        """
        self.net = self.net.to(self.device)
        self.net.eval()
        correct = 0
        total = 0
        torch.cuda.empty_cache()
        L2=0
        Linf=0
        self.confidence_correct=0.0
        self.confidence_incorrect=0.0
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes))
        
        for batch_idx, (data, labels) in enumerate(self.test_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)
            self.current_batch_data['data'] = data
            self.current_batch_data['labels'] = labels
            # Generate adversarial image
            perturbed_data, un_norm_perturbed_data = self.attack.generate_adversary(data, labels, adv_train_model = self.net, targeted=self.targeted, target_class=self.target )
            L2 += torch.sum(torch.norm(data - un_norm_perturbed_data, p=2, dim=(1,2,3)))
            Linf += torch.sum(torch.norm(data - un_norm_perturbed_data, p=float('inf'), dim=(1,2,3)))
            data = self.framework.preprocess(self.framework.normalize(un_norm_perturbed_data)).to(self.device)
            out = self.framework.net(data)
            _, pred = torch.max(out, dim=1)
            sorted_output =  torch.nn.functional.softmax(out, dim=1).sort(dim=1)[0]
            confidence = sorted_output[:,self.num_classes-1] - sorted_output[:,self.num_classes-2]
            correct += (pred == labels).sum().item()
            self.confidence_correct += torch.where(pred == labels, confidence, torch.zeros_like(confidence)).sum().item()
            self.confidence_incorrect += torch.where(pred != labels, confidence, torch.zeros_like(confidence)).sum().item()
            total += labels.size()[0]
            for ground_truth, predicted in zip(labels.cpu(), out.argmax(axis=1).cpu()):
                self.confusion_matrix[ground_truth, predicted] += 1
        if correct !=0:
            self.confidence_correct = self.confidence_correct/correct
            
        if(total != correct):
            self.confidence_incorrect = self.confidence_incorrect/float(total-correct)

        accuracy = float(correct) * 100.0 / float(total)
        norm_2 = float(L2.item())/ float(total)
        norm_inf = float(Linf.item())/ float(total)
        return correct, total, accuracy, norm_2, norm_inf
        
    def test(self):
        """Evaluates network performance and returns the accuracy of the network

        Returns:
            correct, total, accuracy.
            correct (int), the number of correctly classifed images
            total (int), the total number of images in the test set
            accuracy (float), accuracy in %
        """
        self.net = self.net.to(self.device)
        self.net.eval()
        correct = 0
        total = 0
        self.confidence_correct=0.0
        self.confidence_incorrect=0.0
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes))
        torch.cuda.empty_cache()
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.test_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                self.current_batch_data['data'] = data
                self.current_batch_data['labels'] = labels
                # Generate adversarial image
                data = self.preprocess(self.normalize(data))
                out = self.net(data)
                
                _, pred = torch.max(out, dim=1)
                sorted_output =  torch.nn.functional.softmax(out, dim=1).sort(dim=1)[0]
                confidence = sorted_output[:,self.num_classes-1] - sorted_output[:,self.num_classes-2]
                correct += (pred == labels).sum().item()
                self.confidence_correct += torch.where(pred == labels, confidence, torch.zeros_like(confidence)).sum().item()
                self.confidence_incorrect += torch.where(pred != labels, confidence, torch.zeros_like(confidence)).sum().item()
                total += labels.size()[0]
                for ground_truth, predicted in zip(labels.cpu(), out.argmax(axis=1).cpu()):
                    self.confusion_matrix[ground_truth, predicted] += 1
        if correct !=0:
            self.confidence_correct = self.confidence_correct/correct
            
        if(total != correct):
            self.confidence_incorrect = self.confidence_incorrect/float(total-correct)
            
        accuracy = float(correct) * 100.0 / float(total)
        return correct, total, accuracy

    def validate(self):
        """Evaluates network performance on the validation set and returns the accuracy of the network

        Returns:
            correct, total, accuracy, loss
            correct (int), the number of correctly classifed images
            total (int), the total number of images in the test set
            accuracy (float), accuracy in %
            loss (float), average loss over the validation set
        """
        self.net = self.net.to(self.device)
        self.net.eval()
        correct = 0
        total = 0
        running_loss = 0
        batches = 0
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.val_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                self.current_batch_data['data'] = data
                self.current_batch_data['labels'] = labels

                out = self.net(self.preprocess(self.normalize(data)))
                target = self.framework.net( self.framework.preprocess(self.framework.normalize(data)) )
                loss = self.criterion(out,target)

                running_loss += loss.item()
                batches += 1
            
                _, pred = torch.max(out, dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size()[0]
            accuracy = float(correct) * 100.0 / float(total)
            average_loss = running_loss / batches
        return correct, total, accuracy, average_loss

    def train(self,
              resume_training=False,
              batch_hook=None, 
              epoch_hook=None,
              parallel=False,
              parallel_devices_ids=[0,1,2,3],
              visualize = False):
        """Trains the network for the specifed number of epochs
        
        Args:
            resume_training (bool): Resumes traning from previous epoch, optimizer state and net parameters
            batch_hook (function pointer): function called at end of every batch
            epoch_hook (function pointer): function called at end of every epoch
            parallel (bool): Use dataparallel on the devices
            parallel_devices_ids (list of int): the device id's to use
            visualize(bool) : see the image in the dataset.
        """
        self.net = self.net.to(self.device)
        self.net.train()
        if resume_training:
            saved_training_state = torch.load('./pretrained/'+ self.dataset +'/temp/' + self.model_name  + '.temp_tfr')
            start_epoch =  saved_training_state['epoch']
            self.optimizer.load_state_dict(saved_training_state['optimizer'])
            self.net.load_state_dict(saved_training_state['model'])
            self.best_val_accuracy = saved_training_state['best_val_accuracy']
            self.best_val_loss = saved_training_state['best_val_loss']
        else:
            start_epoch = 0
            self.best_val_accuracy = 0.0
            self.best_val_loss = float('inf')

        if parallel:
            self.net = nn.DataParallel(self.net, device_ids=parallel_device_ids)
        else:
            self.net = self.net.to(self.device)

        for epoch in range(start_epoch, self.num_epochs, 1):
            train_correct = 0.0
            train_total = 0.0
            save_ckpt = False
            self.current_epoch = epoch
            
            # Testing on the validation set puts this into eval, so make sure to 
            # put it into train mode each epoch
            self.net.train()

            for batch_idx, (data, labels) in enumerate(self.train_loader):
                self.current_batch = batch_idx

                data = data.to(self.device)
                labels = labels.to(self.device)
                self.current_batch_data['data'] = data
                self.current_batch_data['labels'] = labels
                if visualize:
                    for index in range(10):
                        plt.figure()
                        if self.num_channels==1:
                            plt.imshow(data[index][0].cpu().numpy(),cmap='gray', vmin=0, vmax=1)
                        else:    
                            plt.imshow(data[index].cpu().numpy().transpose(1,2,0))
                        plt.show()
                        plt.close()
                
                # Clears gradients of all the parameter tensors
                self.optimizer.zero_grad()
                out = self.net(self.preprocess(self.normalize(data)))
                target = self.framework.net( self.framework.preprocess(self.framework.normalize(data)) )
                loss = self.criterion(out,target)
                loss.backward()
                self.optimizer.step()

                # Update internal local variables
                train_correct += (out.max(-1)[1] == labels).sum().long().item()
                train_total += labels.size()[0]
                train_accuracy = float(train_correct) * 100.0/float(train_total)

                # Update object attributes accessible outside
                self.current_train_loss = loss.item()
                self.current_train_acc = train_accuracy


                # Call batch hook after batch updates
                if(batch_hook):
                    batch_hook(self)

            if(self.val_split > 0.0): 
                val_correct, val_total, val_accuracy, val_loss = self.validate()
                #Update lr
                self.scheduler.step(val_loss)
            
            
                if val_accuracy >= self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy 
                    self.best_val_loss = val_loss
                    save_ckpt = True
            else: 
                self.scheduler.step()
                val_accuracy = float('inf')
                if (self.current_epoch + 1) % 10 == 0:
                    save_ckpt = True

            if parallel:
                saved_training_state = {
                                            'epoch'             : self.current_epoch + 1,
                                            'optimizer'         : self.optimizer.state_dict(),
                                            'model'             : self.net.module.state_dict(),
                                            'best_val_accuracy' : self.best_val_accuracy,
                                            'best_val_loss'     : self.best_val_loss
                                        }
            else:
                saved_training_state = {
                                            'epoch'             : self.current_epoch + 1,
                                            'optimizer'         : self.optimizer.state_dict(),
                                            'model'             : self.net.state_dict(),
                                            'best_val_accuracy' : self.best_val_accuracy,
                                            'best_val_loss'     : self.best_val_loss
                                        }
            torch.save(saved_training_state, './pretrained/' + self.dataset + '/temp/' + self.model_name + '.temp_tfr')
            
            if save_ckpt:
                self.saves.append(self.current_epoch)
                if parallel:
                    torch.save(self.net.module.state_dict(), './pretrained/'+ self.dataset +'/' + self.model_name  + '.ckpt_tfr')
                else:
                    torch.save(self.net.state_dict(), './pretrained/'+  self.dataset +'/' + self.model_name + '.ckpt_tfr')


                test_correct, test_total, test_accuracy = self.test()
                self.current_test_acc = test_accuracy

            if(epoch_hook):
                epoch_hook(self)

        if(self.num_epochs > 0):
            # Load the most optimum weights found during training
            saved_training_state = torch.load('./pretrained/'+ self.dataset +'/temp/' + self.model_name  + '.temp_tfr')
            self.net.load_state_dict(saved_training_state['model'])
