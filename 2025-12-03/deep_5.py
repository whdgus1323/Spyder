# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 13:51:09 2025

@author: Choe JongHyeon
"""

import numpy as np
import matplotlib.pyplot as plt

x, y = np.meshgrid(
    np.linspace(-2, 2, 3), 
    np.linspace(-3, 3, 3)
)
z = np.exp(-x ** 2 -y ** 2)


print(x)
print(y)
print(z)