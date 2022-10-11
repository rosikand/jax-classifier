"""
File: experiments.py
------------------
This file holds the experiment classes for modularized running. 
"""

import datasets
import models
import torch


class SupervisedExp:
    def __init__(self):
        self.model = models.MLP() 
        self.trainloader, self.testloader = datasets.get_dataloaders(
            path="https://stanford.edu/~rsikand/assets/datasets/mini_cifar.pkl", 
            DatasetClass=datasets.BaseDataset
            )
    
    def perform_init(self):
        """
        Automatically performs model init based on first sample in trainloader. 
        """
        sample_shape = tuple((next(iter(self.trainloader))[0].shape))
        self.model.init_module(sample_shape)
        print("Model initialized successfully!")

    
    def debug(self):
        """
        Testing things by printing to the console. 
        """

        # test shape
        sample = next(iter(self.trainloader))[0].cpu().detach().numpy()
        print(type(sample))
        print(sample.shape)

        preds = self.model.predict(sample)
        print(preds)





