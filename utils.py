import pandas as pd
import torch
from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np


def find_skyline_brute_force(ar: np.array):
    skyline = set()
    size = ar.shape[0]
    for i in range(size):
        dominated = False
        for j in range(size):
            if i == j:
                continue
            if np.all(ar[i] < ar[j]):
                dominated = True
                break
        if not dominated:
            skyline.add(i)
    return skyline