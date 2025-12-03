# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 17:22:18 2025

@author: Choe JongHyeon
"""

# -*- coding: utf-8 -*-
import os
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

net_root = r"C:\Users\Choe JongHyeon\Desktop\Tst_map_v_2"
categories = ["map_1", "map_2", "map_3", "map_4", "map_5"]

grid_rows = 4
grid_cols = 4

density_cdf = {}
lanes_cdf = {}
dist_cdf = {}

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
        continue

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)

    min_x = xs_arr.min()
    max_x = xs_arr.max()
    min_y = ys_arr.min()
    max_y = ys_arr.max()

    dx = (max_x - min_x) / grid_cols if grid_cols > 0 else 0.0
    dy = (max_y - min_y) / grid_rows if grid_rows > 0 else 0.0

    grid_counts = np.zeros((grid_rows, grid_cols), dtype=int)
    for x, y in zip(xs_arr, ys_arr):
        cx = int((x - min_x) / dx) if dx > 0 else 0
        cy = int((y - min_y) / dy) if dy > 0 else 0
        if cx < 0:
            cx = 0
        if cx >= grid_cols:
            cx = grid_cols - 1
        if cy < 0:
            cy = 0
        if cy >= grid_rows:
            cy = grid_rows - 1
        grid_counts[cy, cx] += 1

    dens_vals = grid_counts.flatten().astype(float)
    dens_sorted = np.sort(dens_vals)
    n = len(dens_sorted)
    dens_y = np.arange(1, n + 1) / n
    density_cdf[cat] = (dens_sorted, dens_y)

    incoming = {jid: 0 for jid in junctions.keys()}
    for e in root.iter("edge"):
        func = e.get("function", "")
        if func == "internal":
            continue
        to_id = e.get("to")
        if to_id not in incoming:
            continue
        lane_count = 0
        for ln in e.iter("lane"):
            lane_count += 1
        incoming[to_id] += lane_count

    lane_vals = np.array(list(incoming.values()), dtype=float)
    lane_sorted = np.sort(lane_vals)
    n_lane = len(lane_sorted)
    lane_y = np.arange(1, n_lane + 1) / n_lane
    lanes_cdf[cat] = (lane_sorted, lane_y)

    coords = np.array(list(junctions.values()), dtype=float)
    if coords.shape[0] >= 2:
        dists = []
        for i in range(coords.shape[0]):
            diff = coords - coords[i]
            dist = np.sqrt(np.sum(diff * diff, axis=1))
            mask = dist > 0
            if not np.any(mask):
                continue
            nn = np.min(dist[mask])
            dists.append(nn)
        if dists:
            dists = np.array(dists, dtype=float)
            d_sorted = np.sort(dists)
            n_dist = len(d_sorted)
            dist_y = np.arange(1, n_dist + 1) / n_dist
            dist_cdf[cat] = (d_sorted, dist_y)
    else:
        print(cat, "has less than 2 junctions, skip distance CDF")

plt.figure(figsize=(8, 6))
for cat, (x, y) in density_cdf.items():
    plt.plot(x, y, label=cat)
plt.xlabel("junction count per grid cell")
plt.ylabel("CDF")
plt.title("Junction density CDF (grid-based)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
for cat, (x, y) in lanes_cdf.items():
    plt.plot(x, y, label=cat)
plt.xlabel("incoming lanes per junction")
plt.ylabel("CDF")
plt.title("Incoming-lane CDF (junction-based)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
for cat, (x, y) in dist_cdf.items():
    plt.plot(x, y, label=cat)
plt.xlabel("nearest-neighbor distance between junctions")
plt.ylabel("CDF")
plt.title("Junction nearest-distance CDF")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
