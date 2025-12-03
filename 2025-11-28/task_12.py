# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 10:45:23 2025

@author: Choe JongHyeon
"""

# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

log_root = r"C:\Users\Choe JongHyeon\Desktop\SimpleMap_Test\map"
net_root = r"C:\Users\Choe JongHyeon\Desktop\Tst_map_v_2"

categories = ["map_1", "map_2", "map_3", "map_4", "map_5"]
node_ids = range(1, 21)

grid_rows = 4
grid_cols = 4

log_filename_pos = "position.csv"

congestion_cdf = {}
dist_cdf = {}
degree_cdf = {}

for cat in categories:
    net_dir = cat + "_b_v.2"
    net_path = os.path.join(net_root, net_dir, "map.net.xml")
    if not os.path.exists(net_path):
        print("net not found:", net_path)
        continue

    tree = ET.parse(net_path)
    root = tree.getroot()

    junctions = {}
    xs = []
    ys = []
    for j in root.iter("junction"):
        t = j.get("type", "")
        if t == "internal":
            continue
        jid = j.get("id")
        x = float(j.get("x"))
        y = float(j.get("y"))
        junctions[jid] = (x, y)
        xs.append(x)
        ys.append(y)

    if not junctions:
        print(cat, "no non-internal junctions")
        continue

    xs_arr = np.array(xs, dtype=float)
    ys_arr = np.array(ys, dtype=float)

    min_x = xs_arr.min()
    max_x = xs_arr.max()
    min_y = ys_arr.min()
    max_y = ys_arr.max()

    dx = (max_x - min_x) / grid_cols if grid_cols > 0 else 0.0
    dy = (max_y - min_y) / grid_rows if grid_rows > 0 else 0.0

    grid_junc = np.zeros((grid_rows, grid_cols), dtype=int)
    for x, y in zip(xs_arr, ys_arr):
        if dx > 0:
            cx = int((x - min_x) / dx)
        else:
            cx = 0
        if dy > 0:
            cy = int((y - min_y) / dy)
        else:
            cy = 0
        if cx < 0:
            cx = 0
        if cx >= grid_cols:
            cx = grid_cols - 1
        if cy < 0:
            cy = 0
        if cy >= grid_rows:
            cy = grid_rows - 1
        grid_junc[cy, cx] += 1

    degrees = {jid: 0 for jid in junctions.keys()}
    for e in root.iter("edge"):
        func = e.get("function", "")
        if func == "internal":
            continue
        from_id = e.get("from")
        to_id = e.get("to")
        if from_id in degrees:
            degrees[from_id] += 1
        if to_id in degrees:
            degrees[to_id] += 1

    deg_vals = np.array(list(degrees.values()), dtype=float)
    if len(deg_vals) > 0:
        deg_sorted = np.sort(deg_vals)
        n_deg = len(deg_sorted)
        deg_y = np.arange(1, n_deg + 1) / n_deg
        degree_cdf[cat] = (deg_sorted, deg_y)

    pos_list = []
    for nid in node_ids:
        node_dir = os.path.join(log_root, cat, str(nid), "ART-3.0", "DPC-5")
        pos_path = os.path.join(node_dir, log_filename_pos)
        if not os.path.exists(pos_path):
            continue
        try:
            df = pd.read_csv(pos_path, engine="python")
        except Exception as e:
            print("read_csv failed:", pos_path, "->", e)
            continue
        if not {"Time", "Node", "NodeX", "NodeY"}.issubset(df.columns):
            continue
        pos_list.append(df[["Time", "Node", "NodeX", "NodeY"]])

    if not pos_list:
        print(cat, "no position logs")
        continue

    pos_all = pd.concat(pos_list, ignore_index=True)

    grid_presence = np.zeros((grid_rows, grid_cols), dtype=float)
    coords_junc = np.array(list(junctions.values()), dtype=float)

    dist_vals = []
    for _, row in pos_all.iterrows():
        x = float(row["NodeX"])
        y = float(row["NodeY"])
        if dx > 0:
            cx = int((x - min_x) / dx)
        else:
            cx = 0
        if dy > 0:
            cy = int((y - min_y) / dy)
        else:
            cy = 0
        if cx < 0:
            cx = 0
        if cx >= grid_cols:
            cx = grid_cols - 1
        if cy < 0:
            cy = 0
        if cy >= grid_rows:
            cy = grid_rows - 1
        grid_presence[cy, cx] += 1.0

        dxs = coords_junc[:, 0] - x
        dys = coords_junc[:, 1] - y
        d2 = dxs * dxs + dys * dys
        if d2.size > 0:
            d_min = float(np.sqrt(d2.min()))
            dist_vals.append(d_min)

    cong_vals = []
    rows, cols = grid_junc.shape
    for r in range(rows):
        for c in range(cols):
            jcnt = grid_junc[r, c]
            pcnt = grid_presence[r, c]
            if jcnt > 0 and pcnt > 0:
                cong_vals.append(pcnt / jcnt)

    if cong_vals:
        cong_arr = np.array(cong_vals, dtype=float)
        cong_sorted = np.sort(cong_arr)
        n_cong = len(cong_sorted)
        cong_y = np.arange(1, n_cong + 1) / n_cong
        congestion_cdf[cat] = (cong_sorted, cong_y)

    if dist_vals:
        dist_arr = np.array(dist_vals, dtype=float)
        dist_sorted = np.sort(dist_arr)
        n_dist = len(dist_sorted)
        dist_y = np.arange(1, n_dist + 1) / n_dist
        dist_cdf[cat] = (dist_sorted, dist_y)

    print(cat, "junctions:", len(junctions),
          "deg_samples:", len(deg_vals),
          "congestion_cells:", len(cong_vals),
          "dist_samples:", len(dist_vals))

plt.figure(figsize=(8, 6))
for cat, (x, y) in congestion_cdf.items():
    plt.plot(x, y, label=cat)
plt.xlabel("vehicle presence per junction (grid-level)")
plt.ylabel("CDF")
plt.title("Junction congestion CDF")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
for cat, (x, y) in dist_cdf.items():
    plt.plot(x, y, label=cat)
plt.xlabel("distance from vehicle to nearest junction")
plt.ylabel("CDF")
plt.title("Vehicle–junction distance CDF")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
for cat, (x, y) in degree_cdf.items():
    plt.plot(x, y, label=cat)
plt.xlabel("junction degree (edge connections)")
plt.ylabel("CDF")
plt.title("Junction connectivity (degree) CDF")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
