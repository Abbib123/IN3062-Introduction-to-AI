# -*- coding: utf-8 -*-
"""
Tutorial 1
"""

import sys

import pandas as pd
import sklearn as sk
import numpy as np




arr = np.array([[1, 2, 3, 4, 5], 
                [2, 4, 6, 8, 10], 
                [3, 6, 9, 12, 15],
                [4,8,12,16, 20],
                [5,10,15,20, 25]])


if arr / 2 == 0 :
    print("0")
    
print(arr*arr)
 