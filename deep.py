
#Build in
from os import path

#Installed
import torch
import torchvision
import pandas


class Index():
    def __init__(self, index,
            root=None,
            transformer=None,
            loader=torchvision.datasets.folder.default_loader,
            headers=('img','label'),
            delimiter=' ',
            skiprows=0
            ):
        self.root = root
        self.transformer = transformer
        self.loader = loader
        self.headers = headers
        self._csv = pandas.read_csv(index, delimiter=delimiter, skiprows=skiprows)
    
    @property
    def csv(self):
        return self._csv
    
    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
    
        y = row[self.headers(1)]
        file = row[self.headers(0)]
        
        x = self.loader(path.join(self.root, file))
        if self.transformer:
            x = self.transformer(x)

        return x, y
    
    def __len__(self):
        return self.index.shape[0]


class Classifier():

    def __init__(self, model, optimizer, criterion, device=None, epoch=0, step=0):
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._device = device
        self._epoch=epoch
        self._step=step
        pass
    
    @property
    def model(self):
        return self._model
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @property
    def criterion(self):
        return self._criterion
    
    @property
    def device(self):
        return self._device
    
    @property
    def epoch(self):
        return self._epoch
    
    @property
    def step(self):
        return self._step

    def run(self, dataloader, train=False):
        running_loss = 0.0
        running_corrects = 0
        
        if train:
            classifier.model.train()
        else:
            classifier.model.eval()

        # Iterate over data.
        for step, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(self.model.training):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                # backward + optimize only if in training mode
                if self.model.training:
                    loss.backward()
                    self.optimizer.step()
                    self._step += 1

            # statistics
            step_loss = float(loss.item() * inputs.size(0))
            step_acc = int(torch.sum(preds == labels.data))
            yield outputs, inputs, labels, step, step_loss, step_acc
        
        if self.model.training:
            self._epoch += 1