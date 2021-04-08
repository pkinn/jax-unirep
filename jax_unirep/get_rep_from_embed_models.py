# -*- coding: utf-8 -*-

"""
Models from the original unirep paper.

These models have been edited to work with get_reps. The original get_reps is 
tailored to work with only the 1900, and functions by only applying one layer
to the output of get_embedding. Another method to get reps would be to create 
models without the final mLSTMhiddenstates, Dense, and SoftMax layers.

Generally, you would use these functions in the following fashion:

```python
ADD USAGE HERE
```
"""
from jax.experimental.stax import Dense, Softmax, serial

from .layers import AAEmbedding, mLSTM, mLSTMHiddenStates



def mLSTM1900_gr():
    model_layers = (
        mLSTM(1900),
        )
    init_fun, apply_fun = serial(*model_layers)
    return init_fun, apply_fun

    

def mLSTM256_gr():
    model_layers = (
        mLSTM(256),
        mLSTMHiddenStates(),
        mLSTM(256),
        mLSTMHiddenStates(),
        mLSTM(256),
        mLSTMHiddenStates(),
        mLSTM(256),
    )
    init_fun, apply_fun = serial(*model_layers)
    return init_fun, apply_fun



def mLSTM64_gr():
    
    model_layers = (
        mLSTM(64),
        mLSTMHiddenStates(),
        mLSTM(64),
        mLSTMHiddenStates(),
        mLSTM(64),
        mLSTMHiddenStates(),
        mLSTM(64),
    )
    init_fun, apply_fun = serial(*model_layers)
    return init_fun, apply_fun