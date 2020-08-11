""""""
"""
@authors:  Sangamesh Kodge
@copyright: Nanoelectronics Research Laboratory
"""
""""""
import sys
sys.path.append('../')
###CUDA_VISIBLE_DEVICES=1 python transfer_attack.py --lr=0.001
import torch
from advtrain.framework import Framework
from advtrain.blackbox_attack_extention import Blackbox_extention
from advtrain.instantiate_model import instantiate_model
from advtrain.utils.str2bool import str2bool
import argparse

###CUDA_VISIBLE_DEVICES=0 python train.py --input_quant=nk
parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#training parameters
parser.add_argument('--epochs',                 default=350,                type=int,       help='Set number of epochs')
parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='Set dataset to use')
parser.add_argument('--parallel',               default=False,              type=str2bool,  help='Device in  parallel')
parser.add_argument('--lr',                     default=0.001,               type=float,     help='Learning Rate')
parser.add_argument('--test_accuracy_display',  default=1,                 type=int,       help='Intervals to display test accuracy')
parser.add_argument('--optimizer',              default='SGD',              type=str,       help='Optimizer for training')
parser.add_argument('--loss',                   default='MSE',     type=str,       help='loss function for training')
parser.add_argument('--resume',                 default=False,              type=str2bool,  help='resume training from a saved checkpoint')
#parser.add_argument('--include_validation',     default=False,              type=str2bool,  help='retrains with validation set')

# Dataloader args
parser.add_argument('--train_batch_size',       default=512,    type=int,       help='Train batch size')
parser.add_argument('--test_batch_size',        default=512,    type=int,       help='Test batch size')
parser.add_argument('--val_split',              default=0.1,    type=float,     help='fraction of training dataset split as validation')
parser.add_argument('--augment',                default=True,   type=str2bool,  help='Random horizontal flip and random crop')
parser.add_argument('--padding_crop',           default=4,      type=int,       help='Padding for random crop')
parser.add_argument('--shuffle',                default=True,   type=str2bool,  help='Shuffle the training dataset')
parser.add_argument('--random_seed',            default=None,   type=int,       help='initialising the seed for reproducibility')

# Model parameters
parser.add_argument('--suffix',         default='',         type=str,        help='appended to model name')
parser.add_argument('--arch',           default='resnet',   type=str,        help='Network architecture')
parser.add_argument('--pretrained',     default=True,      type=str2bool,   help='load saved model for ./pretrained/dataset/')
parser.add_argument('--torch_weights',  default=False,      type=str2bool,   help='load torchvison weights for imagenet')
parser.add_argument('--input_quant',    default='FP',       type=str,       help='Quantization transfer function-Q1 Q2 Q4 Q6 Q8 HT FP')
parser.add_argument('--dorefa',         default=False,      type=str2bool,  help='Use Dorefa Net')
parser.add_argument('--qout',           default=False,      type=str2bool,  help='Output layer weight quantisation')
parser.add_argument('--qin',            default=False,      type=str2bool,  help='Input layer weight quantisation')
parser.add_argument('--abit',           default=32,         type=int,       help='activation quantisation precision')
parser.add_argument('--wbit',           default=32,         type=int,       help='Weight quantisation precision')

# Transfer attack model parameters
parser.add_argument('--tfr_suffix',         default='_tfr',         type=str,        help='appended to model name')
parser.add_argument('--tfr_arch',           default='resnet',   type=str,        help='Network architecture')
parser.add_argument('--tfr_pretrained',     default=False,      type=str2bool,   help='load saved model for ./pretrained/dataset/')
parser.add_argument('--tfr_torch_weights',  default=False,      type=str2bool,   help='load torchvison weights for imagenet')
parser.add_argument('--tfr_input_quant',    default='FP',       type=str,       help='Quantization transfer function-Q1 Q2 Q4 Q6 Q8 HT FP')
parser.add_argument('--tfr_dorefa',         default=False,      type=str2bool,  help='Use Dorefa Net')
parser.add_argument('--tfr_qout',           default=False,      type=str2bool,  help='Output layer weight quantisation')
parser.add_argument('--tfr_qin',            default=False,      type=str2bool,  help='Input layer weight quantisation')
parser.add_argument('--tfr_abit',           default=32,         type=int,       help='activation quantisation precision')
parser.add_argument('--tfr_wbit',           default=32,         type=int,       help='Weight quantisation precision')


# Attack parameters
parser.add_argument('--adv_trn',    default=True,      type=str2bool,  help='adv Training')
parser.add_argument('--attack',     default='PGD',      type=str,       help='Type of attack [PGD, CW]')
parser.add_argument('--lib',        default='custom',   type=str,       help='Use [foolbox, advtorch, custom] code for adversarial attack')
parser.add_argument('--use_bpda',   default=True,       type=str2bool,  help='Use Backward Pass through Differential Approximation when using attack')
parser.add_argument('--random',     default=True,       type=str2bool,  help='Random seed/strating points')
parser.add_argument('--iterations', default=40,         type=int,       help='Number of iterations of PGD')
parser.add_argument('--epsilon',    default=0.031,      type=float,     help='epsilon for PGD')
parser.add_argument('--targeted',   default=None,       type=int,       help='Target class for targeted attack, None for non targeted attack')
parser.add_argument('--stepsize',   default=0.01,       type=float,     help='stepsize for attack')

global args
args = parser.parse_args()
print(args)

def epoch_hook(framework):
    if (framework.current_epoch+1)%args.test_accuracy_display==0:
        print("\nEpoch {}: \n Train Loss:{} \n Train Accuracy:{} \n Test Accuracy:{} ".format(framework.current_epoch, framework.current_train_loss, framework.current_train_acc, framework.current_test_acc) )

if args.dataset.lower()=='imagenet':
    num_classes=1000
elif  args.dataset.lower()=='tinyimagenet':
    num_classes=200
elif  args.dataset.lower()=='cifar100':
    num_classes=100
else:
    num_classes=10

net, model_name, Q = instantiate_model(dataset=args.dataset,
                                    num_classes=num_classes,
                                    input_quant=args.input_quant, 
                                    arch=args.arch,
                                    dorefa=args.dorefa,
                                    abit=args.abit,
                                    wbit=args.wbit,
                                    qin=args.qin,
                                    qout=args.qout,
                                    suffix=args.suffix, 
                                    load=args.pretrained,
                                    torch_weights=args.torch_weights,
                                    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


framework = Framework(net=net,
                      model_name=model_name,
                      preprocess=Q,
                      dataset=args.dataset,
                      train_batch_size=args.train_batch_size,
                      test_batch_size=args.test_batch_size,
                      val_split=args.val_split,
                      augment=args.augment,
                      padding_crop=args.padding_crop,
                      shuffle=args.shuffle,
                      random_seed=args.random_seed,
                      adversarial_training=args.adv_trn,
                      lib = args.lib,
                      attack=args.attack,
                      iterations=args.iterations,
                      epsilon=args.epsilon,
                      stepsize=args.stepsize,
                      use_bpda=args.use_bpda,
                      target=args.targeted,
                      random=args.random,
                      device=None)


tfr_net, tfr_model_name, tfr_Q = instantiate_model( dataset=args.dataset,
                                                    num_classes=num_classes,
                                                    input_quant=args.tfr_input_quant, 
                                                    arch=args.tfr_arch,
                                                    dorefa=args.tfr_dorefa,
                                                    abit=args.tfr_abit,
                                                    wbit=args.tfr_wbit,
                                                    qin=args.tfr_qin,
                                                    qout=args.tfr_qout,
                                                    suffix=args.tfr_suffix, 
                                                    load=args.tfr_pretrained,
                                                    torch_weights=args.tfr_torch_weights,
                                                    device= framework.device)

blackbox_extention = Blackbox_extention(framework = framework,
                                        net=tfr_net,
                                        model_name=tfr_model_name,
                                        preprocess=tfr_Q,
                                        epochs=args.epochs,

                                        optimizer=args.optimizer,
                                        loss=args.loss,
                                        learning_rate=args.lr)
_ , _, accuracy = framework.test()
print('\nTest Acc of Teacher model: {}'.format(accuracy))
print("Confidence correct : {} \nConfidence incorrect : {} \nConfusion Matrix:\n{}".format(framework.confidence_correct,framework.confidence_incorrect, framework.confusion_matrix))

if not args.tfr_pretrained:
    blackbox_extention.train(epoch_hook=epoch_hook)
_ , _, accuracy = blackbox_extention.test()
print('\nTest Acc of Student model: {}'.format(accuracy))
print("Confidence correct : {} \nConfidence incorrect : {} \nConfusion Matrix:\n{}".format(blackbox_extention.confidence_correct,blackbox_extention.confidence_incorrect, blackbox_extention.confusion_matrix))


_ , _, accuracy, L2, Linf  = blackbox_extention.adversarial_attack()
print('\nBlackbox attack Acc: {} \nL2  norm: {} \nLinf norm: {}'.format(accuracy,L2, Linf))
print("Confidence correct : {} \nConfidence incorrect : {} \nConfusion Matrix:\n{}".format(blackbox_extention.confidence_correct,blackbox_extention.confidence_incorrect, blackbox_extention.confusion_matrix))
