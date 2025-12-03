# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 16:19:27 2025

@author: Choe JongHyeon
"""
import os
import re
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

log_root = r"C:\Users\Choe JongHyeon\Desktop\SimpleMap_Test\map"
net_root = r"C:\Users\Choe JongHyeon\Desktop\Tst_map_v_2"

categories = ["map_1", "map_2", "map_3", "map_4", "map_5"]
node_ids = range(1, 21)
log_filename_pos = "position.csv"
log_filename_rerr = "rerr_precursor_log.txt"

grid_rows = 4
grid_cols = 4

pattern = re.compile(r"\[RERR 생성\] 시간:\s*([0-9\.]+), 노드:\s*(\S+), Precursor 수:\s*(\d+)")

junction_grid_map = {}
rerr_ratio_map = {}
prec_ratio_map = {}
prec_mean_norm_map = {}

global_junc_max = 0

for cat in categories:
    net_dir = cat + "_b_v.2"
    net_path = os.path.join(net_root, net_dir, "map.net.xml")
    if not os.path.exists(net_path):
        print("net not found:", net_path)
        continue

    tree = ET.parse(net_path)
    root = tree.getroot()

    xs = []
    ys = []
    for j in root.iter("junction"):
        t = j.get("type", "")
        if t == "internal":
            continue
        xs.append(float(j.get("x")))
        ys.append(float(j.get("y")))

    if not xs:
        continue

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    dx = (max_x - min_x) / grid_cols
    dy = (max_y - min_y) / grid_rows

    junc_counts = np.zeros((grid_rows, grid_cols), dtype=int)

    for x, y in zip(xs, ys):
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
        junc_counts[cy, cx] += 1

    rerr_counts = np.zeros((grid_rows, grid_cols), dtype=int)
    prec_sum = np.zeros((grid_rows, grid_cols), dtype=float)
    event_cnt = np.zeros((grid_rows, grid_cols), dtype=int)

    total_events = 0

    for nid in node_ids:
        node_dir = os.path.join(log_root, cat, str(nid), "ART-3.0", "DPC-5")
        pos_path = os.path.join(node_dir, log_filename_pos)
        rerr_path = os.path.join(node_dir, log_filename_rerr)

        if not os.path.exists(pos_path) or not os.path.exists(rerr_path):
            continue

        try:
            pos = pd.read_csv(pos_path, engine="python")
        except Exception as e:
            print("read_csv failed:", pos_path, "->", e)
            continue

        if not {"Time", "Node", "NodeX", "NodeY"}.issubset(pos.columns):
            continue

        pos_index = pos.set_index(["Time", "Node"]).sort_index()

        with open(rerr_path, encoding="utf-8") as f:
            for line in f:
                m = pattern.search(line)
                if not m:
                    continue
                t = float(m.group(1))
                node_name = m.group(2)
                prec = int(m.group(3))

                T = int(t)
                key = (T, node_name)

                try:
                    row = pos_index.loc[key]
                except KeyError:
                    continue

                if isinstance(row, pd.DataFrame):
                    x = float(row["NodeX"].iloc[0])
                    y = float(row["NodeY"].iloc[0])
                else:
                    x = float(row["NodeX"])
                    y = float(row["NodeY"])

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

                rerr_counts[cy, cx] += 1
                prec_sum[cy, cx] += prec
                event_cnt[cy, cx] += 1
                total_events += 1

    prec_mean = np.zeros_like(prec_sum, dtype=float)
    mask = event_cnt > 0
    prec_mean[mask] = prec_sum[mask] / event_cnt[mask]

    total_rerr = rerr_counts.sum()
    if total_rerr > 0:
        rerr_ratio = rerr_counts.astype(float) / float(total_rerr)
    else:
        rerr_ratio = np.zeros_like(rerr_counts, dtype=float)

    total_prec = prec_sum.sum()
    if total_prec > 0:
        prec_ratio = prec_sum.astype(float) / float(total_prec)
    else:
        prec_ratio = np.zeros_like(prec_sum, dtype=float)

    max_prec_mean = prec_mean.max()
    if max_prec_mean > 0:
        prec_mean_norm = prec_mean / max_prec_mean
    else:
        prec_mean_norm = np.zeros_like(prec_mean, dtype=float)

    junction_grid_map[cat] = junc_counts
    rerr_ratio_map[cat] = rerr_ratio
    prec_ratio_map[cat] = prec_ratio
    prec_mean_norm_map[cat] = prec_mean_norm

    jm = junc_counts.max()
    if jm > global_junc_max:
        global_junc_max = jm

    print(cat, "total_events:", int(total_events))

print("global_junc_max:", global_junc_max)

for cat in categories:
    if cat not in junction_grid_map:
        continue

    counts = junction_grid_map[cat]

    data_list = [
        ("RERR ratio (count/total)", rerr_ratio_map.get(cat)),
        ("Precursor ratio (sum/total)", prec_ratio_map.get(cat)),
        ("Normalized precursor mean", prec_mean_norm_map.get(cat)),
    ]

    for title_suffix, data_text in data_list:
        if data_text is None:
            continue

        plt.figure(figsize=(6, 6))
        plt.imshow(counts, origin="lower", vmin=0, vmax=global_junc_max)
        plt.title(f"{cat} junction count (heat) / {title_suffix}", fontsize=12)
        plt.xlabel("grid col")
        plt.ylabel("grid row")
        plt.colorbar()

        rows, cols = counts.shape
        for r in range(rows):
            for c in range(cols):
                heat_val = counts[r, c]
                txt_val = data_text[r, c]
                txt = f"{txt_val:.3f}"
                if global_junc_max > 0 and heat_val > global_junc_max * 0.5:
                    color = "white"
                else:
                    color = "black"
                t = plt.text(c, r, txt, ha="center", va="center", color=color, fontsize=10)
                t.set_path_effects([
                    path_effects.Stroke(linewidth=2, foreground="black"),
                    path_effects.Normal()
                ])

        plt.tight_layout()
        plt.show()
