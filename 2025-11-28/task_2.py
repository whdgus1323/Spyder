# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:01:01 2025

@author: Choe JongHyeon
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

base_dir = r'C:/Users/Choe JongHyeon/Desktop/SimpleMap_Test/map'
categories = ['map_1', 'map_2', 'map_3', 'map_4', 'map_5']
node_ids = [i for i in range(1, 21)]
file_name = 'rerr_precursor_log.txt'

pattern = re.compile(r'Precursor 수:\s*(\d+)')

precursor_table = {}

for cat in categories :
    values = []
    for nid in node_ids :
        file_path = os.path.join (base_dir, cat, str(nid), 'ART-3.0', 'DPC-5', file_name)
        if os.path.exists(file_path) :
            with open(file_path, encoding='utf-8') as f :
                for line in f.readlines() :
                    m = re.search(pattern, line)
                    if m:
                        values.append(m.group(1))
    precursor_table[cat] = values
    
total_index = []
precursor_cdf = {}

precursor_cdf = {}
pre_sort = {}

for cat in categories:
    # 문자열이면 숫자로 변환
    vals = np.array(precursor_table[cat], dtype=int)
    pre_sort[cat] = np.sort(vals)

for cat in categories:
    n = len(pre_sort[cat])
    if n == 0:
        precursor_cdf[cat] = []
        continue

    # index / total
    precursor_cdf[cat] = np.arange(1, n+1) / n

plt.figure(figsize=(10, 7))

for cat in categories:
    if len(pre_sort[cat]) == 0:
        continue

    # step 대신 plot
    plt.plot(pre_sort[cat], precursor_cdf[cat], linewidth=1)

plt.xlabel("Value")
plt.ylabel("CDF")
plt.grid(True)
plt.show()

