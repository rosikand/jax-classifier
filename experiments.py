"""
File: experiments.py
------------------
This file holds the experiment classes for modularized running. 
"""

import datasets
import models
import torch
import jax
import jax.numpy as jnp
import optax 


class MeanMetric:
  """
  Scalar metric designed to use on a per-epoch basis 
  and updated on per-batch basis. For getting average across
  the epoch. 
  """

  def __init__(self):
    self.vals = []

  def update(self, new_val):
    self.vals.append(new_val)

  def reset(self):
    self.vals = []

  def get(self):
    mean_value = sum(self.vals)/len(self.vals)
    return mean_value


class SupervisedExp:
    def __init__(self):
        self.model = models.MLP() 
        self.trainloader, self.testloader = datasets.get_dataloaders(
            path="https://stanford.edu/~rsikand/assets/datasets/mini_cifar.pkl", 
            DatasetClass=datasets.BaseDataset
            )
        self.optimizer = optax.adam(0.001)
        self.opt_state = None
        self.loss_metric = MeanMetric()
        self.losses_per_epoch = []

    def perform_init(self):
        """
        Automatically performs model init based on first sample in trainloader. 
        """
        sample_shape = tuple((next(iter(self.trainloader))[0].shape))
        self.model.init_module(sample_shape)
        print("Model initialized successfully!")

    def loss_fn(self, params, batch):
        """
        Need to define this as a function since we will take the gradient 
        with respect to params. 
        I think I need to pass in model.params since grad needs to take the grad
        with respect to them. 
        """
        sample, label = batch
        self.model.params = params
        preds = self.model.predict(sample)
        oh_label = jax.nn.one_hot(label, num_classes=3)
        loss = jnp.mean(optax.softmax_cross_entropy(preds, oh_label))
        self.loss_metric.update(loss)
        return loss

    def train_step(self, batch):
        grads = jax.grad(self.loss_fn, allow_int=True)(self.model.params, batch)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.model.params)
        self.model.params = optax.apply_updates(self.model.params, updates)

    def train(self, num_epochs):
        """
        Updates self.model.params. 
        """

        # configure optimizer 
        self.opt_state = self.optimizer.init(self.model.params)
        
        # loop 
        for epoch in range(num_epochs):
            for batch in self.trainloader:
                sample = batch[0].cpu().detach().numpy()
                label = batch[1].cpu().detach().item()
                new_batch = (sample, label)
                self.train_step(new_batch)
            epoch_loss = self.loss_metric.get()
            print(epoch_loss)
            self.losses_per_epoch.append(epoch_loss)
            self.loss_metric.reset()


    def debug(self):
        """
        Testing things by printing to the console. 
        """

        # test shape
        batch = next(iter(self.trainloader))
        sample = batch[0].cpu().detach().numpy()
        label = batch[1].cpu().detach().item()
        # print(type(sample))
        # print(label)
        # print(type(label))
        # print(sample.shape)

        # preds = self.model.predict(sample)
        # print(preds)

        # # loss
        # oh_label = jax.nn.one_hot(label, num_classes=3)
        # l = optax.softmax_cross_entropy(preds, oh_label)
        # print(l)

        # test full loss 
        loss = self.loss_fn(self.model.params, (sample, label))
        print(type(loss))
        print(float(loss))









