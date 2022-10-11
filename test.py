import datasets
from experiments import SupervisedExp
import numpy as np
import experiments

trainloader, testloader = datasets.get_dataloaders(
    path="https://stanford.edu/~rsikand/assets/datasets/mini_cifar.pkl", 
    DatasetClass=datasets.BaseDataset
)


shape = tuple((next(iter(trainloader))[0].shape))
print(shape)

exp = experiments.SupervisedExp()

exp.perform_init()
exp.debug()