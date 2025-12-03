# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 17:10:28 2025

@author: Choe JongHyeon
"""

# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

log_root = r"C:\Users\Choe JongHyeon\Desktop\SimpleMap_Test\map"
net_root = r"C:\Users\Choe JongHyeon\Desktop\Tst_map_v_2"

categories = ["map_1", "map_2", "map_3", "map_4", "map_5"]
node_ids = range(1, 21)
log_filename_pos = "position.csv"
log_filename_rerr = "rerr_precursor_log.txt"

grid_rows = 4
grid_cols = 4

pattern = re.compile(r"\[RERR 생성\] 시간:\s*([0-9\.]+), 노드:\s*(\S+), Precursor 수:\s*(\d+)")

non_internal_junctions = {}
total_rerr = {}
total_prec = {}
rerr_per_junc = {}
prec_per_junc = {}

hotspot_junc_pct = {}
hotspot_rerr_pct = {}
hotspot_prec_pct = {}

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

    dx = (max_x - min_x) / grid_cols if grid_cols > 0 else 0.0
    dy = (max_y - min_y) / grid_rows if grid_rows > 0 else 0.0

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

    cnt_junc = int(junc_counts.sum())
    non_internal_junctions[cat] = cnt_junc

    tr = int(rerr_counts.sum())
    tp = float(prec_sum.sum())

    total_rerr[cat] = tr
    total_prec[cat] = tp

    if cnt_junc > 0:
        rerr_per_junc[cat] = tr / cnt_junc
        prec_per_junc[cat] = tp / cnt_junc
    else:
        rerr_per_junc[cat] = 0.0
        prec_per_junc[cat] = 0.0

    cells = grid_rows * grid_cols

    j_mean = float(junc_counts.mean())
    if j_mean > 0:
        hj = np.sum(junc_counts > 1.5 * j_mean)
        hotspot_junc_pct[cat] = 100.0 * hj / cells
    else:
        hotspot_junc_pct[cat] = 0.0

    r_mean = float(rerr_counts.mean())
    if r_mean > 0:
        hr = np.sum(rerr_counts > 1.5 * r_mean)
        hotspot_rerr_pct[cat] = 100.0 * hr / cells
    else:
        hotspot_rerr_pct[cat] = 0.0

    p_mean = float(prec_sum.mean())
    if p_mean > 0:
        hp = np.sum(prec_sum > 1.5 * p_mean)
        hotspot_prec_pct[cat] = 100.0 * hp / cells
    else:
        hotspot_prec_pct[cat] = 0.0

print("map  junctions  total_rerr  total_prec  rerr_per_junc  prec_per_junc  hot_junc%  hot_rerr%  hot_prec%")
for cat in categories:
    if cat not in non_internal_junctions:
        continue
    jv = non_internal_junctions.get(cat, 0)
    tr = total_rerr.get(cat, 0)
    tp = total_prec.get(cat, 0.0)
    rp = rerr_per_junc.get(cat, 0.0)
    pp = prec_per_junc.get(cat, 0.0)
    hj = hotspot_junc_pct.get(cat, 0.0)
    hr = hotspot_rerr_pct.get(cat, 0.0)
    hp = hotspot_prec_pct.get(cat, 0.0)
    print(
        cat,
        jv,
        tr,
        f"{tp:.1f}",
        f"{rp:.4f}",
        f"{pp:.4f}",
        f"{hj:.1f}",
        f"{hr:.1f}",
        f"{hp:.1f}"
    )

cats = [cat for cat in categories if cat in non_internal_junctions]

if cats:
    x = np.arange(len(cats))

    rvals = [rerr_per_junc.get(cat, 0.0) for cat in cats]
    pvals = [prec_per_junc.get(cat, 0.0) for cat in cats]

    plt.figure(figsize=(6, 4))
    plt.bar(x, rvals)
    plt.xticks(x, cats, rotation=45)
    plt.ylabel("RERR per junction")
    plt.title("RERR per junction per map")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(x, pvals)
    plt.xticks(x, cats, rotation=45)
    plt.ylabel("Precursor per junction")
    plt.title("Precursor per junction per map")
    plt.tight_layout()
    plt.show()

    hj_vals = [hotspot_junc_pct.get(cat, 0.0) for cat in cats]
    hr_vals = [hotspot_rerr_pct.get(cat, 0.0) for cat in cats]
    hp_vals = [hotspot_prec_pct.get(cat, 0.0) for cat in cats]

    plt.figure(figsize=(6, 4))
    plt.bar(x, hj_vals)
    plt.xticks(x, cats, rotation=45)
    plt.ylabel("hotspot junction cells [%]")
    plt.title("junction hotspot cell ratio per map")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(x, hr_vals)
    plt.xticks(x, cats, rotation=45)
    plt.ylabel("hotspot RERR cells [%]")
    plt.title("RERR hotspot cell ratio per map")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(x, hp_vals)
    plt.xticks(x, cats, rotation=45)
    plt.ylabel("hotspot precursor cells [%]")
    plt.title("precursor hotspot cell ratio per map")
    plt.tight_layout()
    plt.show()
