# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 13:23:46 2025

@author: Choe JongHyeon
"""

import numpy as np
import matplotlib.pyplot as plt

x, y = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-1.3, 1.3, 10))
z = np.exp(-x ** 2 -y ** 2)

plt.contourf(x, y, z, cmap='viridis')
plt.colorbar()  # 색상 바 추가
plt.title('Contourf Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
 
plt.show()