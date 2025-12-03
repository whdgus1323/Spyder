# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 16:12:38 2025

@author: Choe JongHyeon
"""

import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np

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

for md in map_dirs:
    xml_path = os.path.join(base_root, md, xml_name)
    if not os.path.exists(xml_path):
        print("not found:", xml_path)
        continue

    tree = ET.parse(xml_path)
    root = tree.getroot()

    xs = []
    ys = []
    junctions = []

    for j in root.iter("junction"):
        t = j.get("type", "")
        if t == "internal":
            continue
        x = float(j.get("x"))
        y = float(j.get("y"))
        jid = j.get("id")
        xs.append(x)
        ys.append(y)
        junctions.append((jid, x, y))

    if not junctions:
        continue

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    dx = (max_x - min_x) / grid_cols if grid_cols > 0 else 0.0
    dy = (max_y - min_y) / grid_rows if grid_rows > 0 else 0.0

    cell_junctions = defaultdict(list)

    for jid, x, y in junctions:
        if dx > 0:
            cx = int((x - min_x) / dx)
        else:
            cx = 0
        if dy > 0:
            cy = int((y - min_y) / dy)
        else:
            cy = 0

        if cx == grid_cols:
            cx = grid_cols - 1
        if cy == grid_rows:
            cy = grid_rows - 1

        cell_junctions[(cy, cx)].append(jid)

    grid_data[md] = {
        "bounds": (min_x, max_x, min_y, max_y),
        "cell_junctions": cell_junctions,
    }

print("=== GRID JUNCTION STATS ===")
for md in map_dirs:
    if md not in grid_data:
        continue
    print(f"\n[{md}]")
    min_x, max_x, min_y, max_y = grid_data[md]["bounds"]
    print("bounds x:", min_x, "->", max_x)
    print("bounds y:", min_y, "->", max_y)
    cell_junctions = grid_data[md]["cell_junctions"]
    for r in range(grid_rows):
        for c in range(grid_cols):
            key = (r, c)
            cnt = len(cell_junctions.get(key, []))
            print(f"cell({r},{c}) junction_count = {cnt}")
