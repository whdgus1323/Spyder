# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 16:15:58 2025

@author: Choe JongHyeon
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

base_root = r"C:\Users\Choe JongHyeon\Desktop\Tst_map_v_2"
map_dirs = [
    "map_1_b_v.2",
    "map_2_b_v.2",
    "map_3_b_v.2",
    "map_4_b_v.2",
    "map_5_b_v.2",
]
xml_name = "map.net.xml"

grid_rows = 4
grid_cols = 4

grid_data = {}
global_max = 0
global_min = 99999999

for md in map_dirs:
    xml_path = os.path.join(base_root, md, xml_name)
    if not os.path.exists(xml_path):
        print("not found:", xml_path)
        continue

    tree = ET.parse(xml_path)
    root = tree.getroot()

    xs = []
    ys = []

    for j in root.iter("junction"):
        t = j.get("type", "")
        if t == "internal":
            continue
        x = float(j.get("x"))
        y = float(j.get("y"))
        xs.append(x)
        ys.append(y)

    if not xs:
        continue

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    dx = (max_x - min_x) / grid_cols
    dy = (max_y - min_y) / grid_rows

    counts = np.zeros((grid_rows, grid_cols), dtype=int)

    for x, y in zip(xs, ys):
        cx = int((x - min_x) / dx) if dx > 0 else 0
        cy = int((y - min_y) / dy) if dy > 0 else 0
        if cx == grid_cols:
            cx = grid_cols - 1
        if cy == grid_rows:
            cy = grid_rows - 1
        counts[cy, cx] += 1

    grid_data[md] = counts
    local_max = counts.max()
    local_min = counts.min()

    if local_max > global_max:
        global_max = local_max
    if local_min < global_min:
        global_min = local_min

for md in map_dirs:
    if md not in grid_data:
        continue

    counts = grid_data[md]

    plt.figure(figsize=(6, 6))
    plt.imshow(counts, origin="lower", vmin=global_min, vmax=global_max)
    plt.title(f"{md} grid heatmap")
    plt.xlabel("grid col")
    plt.ylabel("grid row")
    plt.colorbar()

    for r in range(grid_rows):
        for c in range(grid_cols):
            value = counts[r, c]
            plt.text(c, r, str(value), ha="center", va="center", color="white" if value > (global_max/2) else "black")

    plt.show()

