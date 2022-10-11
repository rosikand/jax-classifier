"""
File: experiments.py
------------------
Holds Stax modules. 
"""

import jax
import jax.numpy as jnp
from jax import random
from jax.example_libraries import stax
from jax.example_libraries.stax import (
    Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax, Softmax
)

# Use stax to set up network initialization and evaluation functions

class MLP:
    """
    Simple Stax MLP module with init wrapper. 
    """   
    def __init__(self, key=None):
        self.net_init, self.net_apply = stax.serial(
            Flatten,
            Dense(128), Relu,
            Dense(64), Relu,
            Dense(3), Softmax,
        )
        if key is None:
            self.key = random.PRNGKey(42)

        self.params = None
    
    def init_module(self, input_shape):
        out_shape, net_params = self.net_init(self.key, input_shape)
        self.params = net_params
    
    def predict(self, inputs):
        if self.params is None:
            raise Exception("Must call init_module first.")
        preds = self.net_apply(self.params, inputs)
        return preds

