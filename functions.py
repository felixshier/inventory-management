# imports
import numpy as np

# functions

# quantization function

def quantization(val, binSize):
    newVal = binSize * np.round(val/binSize)
    return newVal