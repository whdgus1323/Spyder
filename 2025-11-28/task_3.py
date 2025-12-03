# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 15:37:54 2025

@author: Choe JongHyeon
"""

# -*- coding: utf-8 -*-

import os
import xml.etree.ElementTree as ET
from collections import Counter
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

map_stats = {}

for md in map_dirs:
    xml_path = os.path.join(base_root, md, xml_name)
    if not os.path.exists(xml_path):
        print("not found:", xml_path)
        continue

    tree = ET.parse(xml_path)
    root = tree.getroot()

    total_junctions = 0
    non_internal_junctions = 0
    type_counter = Counter()
    degrees = []

    for j in root.iter("junction"):
        total_junctions += 1
        t = j.get("type", "")
        type_counter[t] += 1
        if t != "internal":
            non_internal_junctions += 1
            inc = j.get("incLanes", "").strip()
            if inc:
                degrees.append(len(inc.split()))

    avg_deg = float(np.mean(degrees)) if degrees else 0.0

    map_stats[md] = {
        "total_junctions": total_junctions,
        "non_internal_junctions": non_internal_junctions,
        "avg_incoming_lanes": avg_deg,
        "type_counts": dict(type_counter),
    }

print("=== MAP JUNCTION STATS ===")
for md in map_dirs:
    if md not in map_stats:
        continue
    st = map_stats[md]
    print(f"\n[{md}]")
    print(f"  total_junctions                : {st['total_junctions']}")
    print(f"  non_internal_junctions         : {st['non_internal_junctions']}")
    print(f"  avg_incoming_lanes (non-int.)  : {st['avg_incoming_lanes']:.3f}")
    print(f"  type_counts:")
    for t, c in st["type_counts"].items():
        print(f"    - {t:12s}: {c}")

labels = []
non_internal_list = []

for md in map_dirs:
    if md not in map_stats:
        continue
    labels.append(md.replace("_b_v.2", ""))
    non_internal_list.append(map_stats[md]["non_internal_junctions"])

plt.figure(figsize=(10, 6))
plt.plot(labels, non_internal_list, marker="o")
plt.xlabel("map")
plt.ylabel("non-internal junction count")
plt.grid(True)
plt.show()


