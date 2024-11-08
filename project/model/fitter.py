# Main Fitter class
# skeletal structure of Fitter class
# override and add methods as needed

import torch

class Fitter:
    def __init__(self, config, logger=None):
        self.model = config['model']
        self.optimizer = config['optimizer'](self.model.parameters(), lr=config['lr'], eps=1e-4)
        self.device = config['device']
        self.epochs = config['epochs']
        self.loss = config['loss']
        self.logger = logger

    def log(self, s):
        if self.logger:
            self.logger.info(s)
        else:
            print(s)

    def fit(self, train_loader, val_loader):
        # keep track of history
        train_history = []
        val_history = []

        # loop with self.epochs
        for epoch in range(self.epochs):
            self.log(f'EPOCH {epoch} ==============================')

            # train with self.train_one_epoch
            train_loss = self.train_one_epoch(train_loader)

            # validate
            val_loss = self.validation(val_loader)

            # add losses to histories
            train_history.append(train_loss)
            val_history.append(val_loss)

            # if epoch % 100 == 0:
            #     print(f"Epoch: {epoch}\tLoss: {train_loss}\tTest Loss: {val_loss}")

        return train_history, val_history

    def train_one_epoch(self, train_loader):
        self.model.train()
        # train model
        pass

    def validation(self, val_loader):
        self.model.eval()
        with torch.inference_mode():
            # conduct evaluation
            pass
        pass
