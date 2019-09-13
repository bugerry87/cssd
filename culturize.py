#!/usr/bin/env python3


#BuildIn
import os
from argparse import ArgumentParser

#Installed
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
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
		description='Demo for Classifying Handwritten Digits 0-4 via SVM',
		parents=parents
		)
	
	parser.add_argument(
		'--data_dir', '-X',
		help='A folder of class wise separated images',
		default='data'
		)
    
    parser.add_argument(
		'--device', '-D',
		help='Force device selection',
		default=None
		)
    
    parser.add_argument(
		'--save_dir', '-S',
		help='A folder of class wise separated images',
		default=None
		)
    
    parser.add_argument(
		'--load', '-L',
		help='A folder of class wise separated images',
		default=None
		)

	return parser


def val_known_args(args):
    if not args.device:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    return args


def create_dataloaders(args):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
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

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x]) for x in args.phases}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in args.phases}
    return dataloaders


def run_epoch(classifier, dataloaders, scheduler, phases):
    for phase in phases:
        if phase == 'train':
            classifier.model.train()
        else:
            classifier.model.eval()
    
        for outputs, inputs, labels, step, step_loss, step_acc in classifier.run(dataloaders[phase]):
            running_loss += step_loss
            running_acc += step_acc
            print('Step: {} Loss: {:.4f} Acc: {:.4f}'.format(step, step_loss, step_acc))

        phase_loss = running_loss / dataset_sizes[phase]
        phase_acc = running_acc.double() / dataset_sizes[phase]
        
        if phase == 'train':
            scheduler.step()

        print('-' * 10)
        print('Phase: {} Loss: {:.4f} Acc: {:.4f}'.format(phase, phase_loss, phase_acc))
        print()
        yield phase_acc, phase_loss, phase


def main(args):
    device = torch.device(args.device)
    
    if not args.load:
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(device)
    elif os.path.isfile(args.load):
        checkpoint = torch.load(args.load)
        
    else:
        raise ValueError("{} is not a file!".format(args.load))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    classifier = Classifier(model, optimizer, criterion, device)
    dataloaders = create_dataloaders(args)
    scheduler = lr_scheduler.StepLR(classifier.optimizer, step_size=args.decay, gamma=args.gamma) 
    
    watch = Stopwatch()
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print()
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('-' * 10)
 
        for phase_acc, phase_loss, phase in run_epoch(classifier, dataloaders, scheduler, args.phases):
            if phase == 'val' and phase_acc > best_acc:
                best_acc = phase_acc
                best_model = copy.deepcopy(classifier.model.state_dict())
                
                if os.path.isdir(args.save_dir):
                    torch.save(best_model, os.path.join(args.save_dir, args.name + '_best.model'))
        
    elapsed = watch.elapsed()
    print('Complete in {:.0f}h {:.0f}m {:.0f}s'.format(elapsed // 360,elapsed // 60, elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return 0


if __name__ == '__main__':
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    args = val_known_args(args)
    main(args)
