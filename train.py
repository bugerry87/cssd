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
from deep import Classifier
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
        '--data', '-X',
        help='A folder of class wise separated images',
        default='data'
        )
    
    parser.add_argument(
        '--device', '-D',
        help='Force device selection',
        default=None
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
        '--phases', '-p',
        help='Phases per epoch',
        default=('train', 'val')
        )

    return parser


def val_known_args(args):
    if not args.device:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    return args


def create_dataloaders(args):
    # Data augmentation and normalization for training
    # Just normalization for validation
    transformer = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data, x), transformer[x]) for x in args.phases}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in args.phases}
    return dataloaders


def run_epoch(classifier, dataloaders, scheduler, phases):
    running_loss = 0
    running_acc = 0

    for phase in phases:
        if phase == 'train':
            classifier.model.train()
        else:
            classifier.model.eval()
    
        for outputs, inputs, labels, step, step_loss, step_acc in classifier.run(dataloaders[phase]):
            running_loss += step_loss
            running_acc += step_acc
            print('Step: {} Loss: {:.4f} Acc: {}/{}'.format(step, step_loss, step_acc, len(outputs)))

        phase_loss = running_loss / step
        phase_acc = running_acc.double() / step
        
        if phase == 'train':
            scheduler.step()

        print('-' * 10)
        print('Phase: {} Loss: {:.4f} Acc: {:.4f}'.format(phase, phase_loss, phase_acc))
        print()
        yield phase_acc, phase_loss, phase
        
        
def load_checkpoint(args):
    device = torch.device(args.device)
    
    if not args.load:
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay, gamma=args.gamma)
        epoch = 0
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
    else:
        raise ValueError("{} is not a file!".format(args.load))

    criterion = nn.CrossEntropyLoss()
    return Classifier(model, optimizer, criterion, device), scheduler, epoch


def main(args):
    classifier, scheduler, epoch = load_checkpoint(args)
    dataloaders = create_dataloaders(args)
      
    watch = Stopwatch()
    best_acc = 0.0
    
    for epoch in range(epoch, args.epochs):
        print()
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('-' * 10)
 
        for phase_acc, phase_loss, phase in run_epoch(classifier, dataloaders, scheduler, args.phases):
            if phase == 'val' and phase_acc > best_acc and os.path.isdir(args.export):
                best_acc = phase_acc
                torch.save(classifier.model.state_dict(), os.path.join(args.export, args.name + '.model'))
            
            if phase == 'train' and os.path.isdir(args.save):
                torch.save({
                    'model': classifier.model.state_dict(),
                    'optimizer': classifier.optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch
                },
                os.path.join(args.export, '{}_{:0>4}.chkpnt'.format(args.name, epoch+1)))
        
    elapsed = watch.elapsed()
    print('Complete in {:.0f}h {:.0f}m {:.0f}s'.format(elapsed // 360,elapsed // 60, elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return 0


if __name__ == '__main__':
    parser = init_arg_parser()
    args, _ = parser.parse_known_args()
    args = val_known_args(args)
    main(args)
