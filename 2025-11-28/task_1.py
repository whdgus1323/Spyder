# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 12:36:58 2025

@author: Choe JongHyeon
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

base_root = r"C:\Users\Choe JongHyeon\Desktop\SimpleMap_Test\map"
categories = ['map_1', 'map_2', 'map_3', 'map_4', 'map_5']
node_ids = range(1, 21)
filename = 'rerr_precursor_log.txt'

pattern = re.compile (r'Precursor 수:\s*(\d+)')
category_cdf = {}

for category in categories :
    values = []
    
    for nid in node_ids:
        dir_path = os.path.join(base_root, category, str(nid), 'ART-3.0', 'DPC-5')
        file_path = os.path.join(dir_path, filename)
        if not os.path.exists(file_path) :
            continue
        
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    values.append(int(m.group(1)))
    
    if len(values) == 0:
        continue
    
    counter = Counter(values)
    sorted_keys = sorted(counter.keys())
    counts = np.array([counter[k] for k in sorted_keys], dtype=float)
    total = counts.sum()
    
    cdf = np.cumsum(counts) / total
    category_cdf[category] = (sorted_keys, cdf)

plt.figure(figsize=(10, 6))

for category, (x, y) in category_cdf.items() :
    plt.plot(x, y, label=category)
    
plt.xlabel('Precursor Value')
plt.ylabel('CDF (Cumulative Ratio)')
plt.title('Precursor CDF Comparison')
plt.grid(True)
plt.legend()
plt.show()