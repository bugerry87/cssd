

class Classifier():

    def __init__(self, model, optimizer, criterion, device=None):
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._device = device
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

    def run(self, dataloader):
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for step, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()

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

            # statistics
            step_loss = loss.item() * inputs.size(0)
            step_acc = torch.sum(preds == labels.data)
            yield outputs, inputs, labels, step, step_loss, step_acc
    