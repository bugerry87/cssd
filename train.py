#!/usr/bin/env python3

#BuildIn
import os
from argparse import ArgumentParser

#Installed
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

#Local
from deep import Classifier, Index
from utils import *


def init_arg_parser(parents=[]):
    '''
    Initialize an ArgumentParser for this script.
    
    Args:
        parents: A list of ArgumentParsers of other scripts, if there are any.
        
    Returns:
        parser: The ArgumentParsers.
    '''
    parser = ArgumentParser(
        description='Train a classifier',
        parents=parents
        )
    
    parser.add_argument(
        '--train', '-X',
        help='An index file of training samples',
        default=None
        )
    
    parser.add_argument(
        '--val', '-Y',
        help='An index file of validation samples',
        default=None
        )
    
    parser.add_argument(
        '--root', '-R',
        help='Root dir',
        default=None
        )
    
    parser.add_argument(
        '--device', '-D',
        help='Force device selection. -1 = cpu',
        type=int,
        default=0
        )
    
    parser.add_argument(
        '--save', '-S',
        help='A folder for the checkpoints',
        default='model'
        )
    
    parser.add_argument(
        '--export', '-E',
        help='A folder for the fix model export',
        default='model'
        )
    
    parser.add_argument(
        '--name', '-N',
        help='A name for the model',
        default='cssd'
        )
    
    parser.add_argument(
        '--load', '-L',
        help='A certain checkpoint to proceed at',
        default=None
        )
    
    parser.add_argument(
        '--epochs', '-e',
        help='Number of epochs',
        type=int,
        default=25
        )
    
    parser.add_argument(
        '--lr', '-l',
        help='The learning rate',
        type=float,
        default=0.001
        )
    
    parser.add_argument(
        '--mom', '-m',
        help='Momentum',
        type=float,
        default=0.9
        )
    
    parser.add_argument(
        '--decay', '-d',
        help='Epochs before learning rate decays',
        type=int,
        default=7
        )
    
    parser.add_argument(
        '--gamma', '-g',
        help='The ratio of the learning rate decay',
        type=float,
        default=0.1
        )
    
    parser.add_argument(
        '--batch', '-b',
        help='The batch size',
        type=int,
        default=4
        )

    return parser


def val_known_args(args):
    return args


def create_dataloaders(args):
    if args.train:
        trainies = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainies = Index(args.train, args.root, trainies)
        trainies = torch.utils.data.DataLoader(trainies, batch_size=args.batch, shuffle=True, num_workers=args.batch)
    else:
        trainies = None
    

    if args.val:
        valies = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        valies = Index(args.val, args.root, valies)
        valies = torch.utils.data.DataLoader(valies, batch_size=args.batch, shuffle=True, num_workers=args.batch)
    else:
        valies = None
    
    return trainies, valies


def run_epoch(classifier, dataloader, train=False):
    epoch_loss = 0
    epoch_acc = 0

    for outputs, inputs, labels, step, step_loss, step_acc in classifier.run(dataloader, train=train):
        epoch_loss += step_loss
        epoch_acc += step_acc
        print('{}\t Step: {} Loss: {:.4f} Acc: {}/{}'.format(step, classifier.step, step_loss, step_acc, len(outputs)))

    epoch_loss = epoch_loss / step
    epoch_acc = float(epoch_acc) / step
    
    print('-' * 10)
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train' if train else 'Val',epoch_loss, epoch_acc))
    print()
    
    return epoch_loss, epoch_acc, step
    

def load_checkpoint(args):
    if args.device is -1:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        device = torch.cuda.current_device()
        print("Switch to device {}".format(device))
    else:
        device = torch.device('cpu')
        print("Warning: No CUDA, fallback to cpu")
    
    if not args.load:
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay, gamma=args.gamma)
        epoch = 0
        step = 0
    elif os.path.isfile(args.load):
        checkpoint = torch.load(args.load)
        model = models.resnet18(pretrained=False)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay, gamma=args.gamma)
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        step = checkpoint['step']
    else:
        raise ValueError("{} is not a file!".format(args.load))

    criterion = nn.CrossEntropyLoss()
    return Classifier(model, optimizer, criterion, device, epoch, step), scheduler


def main(args):
    classifier, scheduler = load_checkpoint(args)
    trainies, valies = create_dataloaders(args)
      
    watch = Stopwatch()
    best_acc = 0.0
    
    if trainies:
        for epoch in range(classifier.epoch, args.epochs):
            print()
            print('Epoch {}/{}'.format(epoch+1, args.epochs))
            print('-' * 10)
     
            run_epoch(classifier, trainies, True)
            scheduler.step()
            
            if os.path.isdir(args.save):
                torch.save({
                    'model': classifier.model.state_dict(),
                    'optimizer': classifier.optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': classifier.epoch,
                    'step': classifier.step
                },
                os.path.join(args.export, '{}_{:0>4}.chkpnt'.format(args.name, epoch+1)))
            
            if valies:
                _, epoch_acc, _ = run_epoch(classifier, valies, False)
                if epoch_acc > best_acc and os.path.isdir(args.export):
                    best_acc = epoch_acc
                    torch.save(classifier.model.state_dict(), os.path.join(args.export, args.name + '.model'))
    
    elif valies:
        run_epoch(classifier, valies, False)
    else:
        raise ValueError("Neither training nor validation data is given!")
        
    elapsed = watch.elapsed()
    print('Complete in {:.0f}h {:.0f}m {:.0f}s'.format(elapsed // 360,elapsed // 60, elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return 0


if __name__ == '__main__':
    parser = init_arg_parser()
    args, _ = parser.parse_known_args()
    args = val_known_args(args)
    main(args)
