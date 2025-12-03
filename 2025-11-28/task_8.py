# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 18:xx:xx 2025

@author: Choe JongHyeon
"""
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
log_filename_rerr = "rerr_precursor_log.txt"

pattern_prec = re.compile(r"Precursor 수:\s*(\d+)")
pattern_time = re.compile(r"시간:\s*([0-9\.]+)")
pattern_node = re.compile(r"노드:\s*(\S+)")

t = pd.DataFrame()

non_internal_junctions = {}
num_rerr = {}
sum_precursor = {}
avg_precursor = {}

for cat in categories:
    net_dir = cat + "_b_v.2"
    net_path = os.path.join(net_root, net_dir, "map.net.xml")
    if not os.path.exists(net_path):
        print("net not found:", net_path)
        continue

    tree = ET.parse(net_path)
    root = tree.getroot()

    cnt_junc = 0
    for j in root.iter("junction"):
        t = j.get("type", "")
        if t == "internal":
            continue
        cnt_junc += 1

    non_internal_junctions[cat] = cnt_junc

    rerr_list = []

    for nid in node_ids:
        node_dir = os.path.join(log_root, cat, str(nid), "ART-3.0", "DPC-5")
        rerr_path = os.path.join(node_dir, log_filename_rerr)
        if not os.path.exists(rerr_path):
            continue

        with open(rerr_path, encoding="utf-8") as f:
            for line in f:
                m_prec = pattern_prec.search(line)
                if not m_prec:
                    continue
                prec = int(m_prec.group(1))
                rerr_list.append(prec)

    if len(rerr_list) == 0:
        num_rerr[cat] = 0
        sum_precursor[cat] = 0
        avg_precursor[cat] = 0.0
    else:
        num_rerr[cat] = len(rerr_list)
        sum_precursor[cat] = int(np.sum(rerr_list))
        avg_precursor[cat] = float(np.mean(rerr_list))

print("map  non_internal_junctions  num_rerr  sum_precursor  avg_precursor")
for cat in categories:
    if cat not in non_internal_junctions:
        continue
    print(
        cat,
        non_internal_junctions.get(cat, 0),
        num_rerr.get(cat, 0),
        sum_precursor.get(cat, 0),
        f"{avg_precursor.get(cat, 0.0):.3f}"
    )

cats = []
junc_vals = []
avg_prec_vals = []

for cat in categories:
    if cat not in non_internal_junctions:
        continue
    cats.append(cat)
    junc_vals.append(non_internal_junctions.get(cat, 0))
    avg_prec_vals.append(avg_precursor.get(cat, 0.0))

if len(cats) > 0:
    x = np.arange(len(cats))

    plt.figure(figsize=(6, 4))
    plt.bar(x, junc_vals)
    plt.xticks(x, cats, rotation=45)
    plt.ylabel("non-internal junctions")
    plt.title("non-internal junction count per map")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(x, avg_prec_vals)
    plt.xticks(x, cats, rotation=45)
    plt.ylabel("avg precursor per RERR")
    plt.title("average precursor per RERR per map")
    plt.tight_layout()
    plt.show()
