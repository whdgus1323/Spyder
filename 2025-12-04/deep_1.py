# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 1) CSV로 지형 데이터 만들기
x = np.linspace(-6, 6, 300)
y = np.linspace(-6, 6, 300)
X, Y = np.meshgrid(x, y)

Z = (
    4.0 * np.exp(-0.25 * ((X + 1.5)**2 + (Y + 0.5)**2))
    + 3.0 * np.exp(-0.35 * ((X - 2.5)**2 + (Y - 2.0)**2))
    + 2.5 * np.exp(-0.4  * ((X + 3.0)**2 + (Y - 3.0)**2))
    - 3.5 * np.exp(-0.3  * ((X - 0.5)**2 + (Y + 3.5)**2))
    - 2.0 * np.exp(-0.5  * ((X + 3.5)**2 + (Y + 2.5)**2))
    + 0.5 * np.sin(2 * X) * np.cos(2 * Y)
    + 0.3 * np.sin(3 * np.sqrt(X**2 + Y**2))
)

df = pd.DataFrame({
    "x": X.ravel(),
    "y": Y.ravel(),
    "z": Z.ravel()
})

csv_path = "terrain_data.csv"
df.to_csv(csv_path, index=False)

# 2) CSV를 입력 데이터로 읽어서 3D + 레이더 구역화

df_in = pd.read_csv(csv_path)

x_vals = np.sort(df_in["x"].unique())
y_vals = np.sort(df_in["y"].unique())
nx = len(x_vals)
ny = len(y_vals)

Xg, Yg = np.meshgrid(x_vals, y_vals)
Zg = df_in["z"].values.reshape(ny, nx)

r = np.sqrt(Xg**2 + Yg**2)
theta = np.arctan2(Yg, Xg)

r_flat = r.ravel()
theta_flat = theta.ravel()
z_flat = Zg.ravel()

n_r = 20
n_theta = 36
r_edges = np.linspace(0, r_flat.max(), n_r + 1)
theta_edges = np.linspace(-np.pi, np.pi, n_theta + 1)

r_idx = np.digitize(r_flat, r_edges) - 1
theta_idx = np.digitize(theta_flat, theta_edges) - 1

sector_vals = np.full((n_theta, n_r), np.nan)
for i in range(n_theta):
    for j in range(n_r):
        mask = (theta_idx == i) & (r_idx == j)
        if np.any(mask):
            sector_vals[i, j] = z_flat[mask].mean()

Theta, R = np.meshgrid(theta_edges, r_edges, indexing="ij")

fig = plt.figure(figsize=(14, 6))

ax3d = fig.add_subplot(121, projection='3d')
ax3d.contour3D(Xg, Yg, Zg, 200, cmap='winter')
ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Height")
ax3d.set_title("3D Blue Mountain")
ax3d.view_init(elev=50, azim=140)

ax2d = fig.add_subplot(122, projection='polar')
c = ax2d.pcolormesh(Theta, R, sector_vals, cmap='winter', shading='auto')

step = 40
ax2d.scatter(theta_flat[::step], r_flat[::step], c=z_flat[::step], s=5, cmap='winter', edgecolors='none')

ax2d.set_title("Radar-style Z Zones")
ax2d.set_rmax(r_flat.max())

plt.tight_layout()
plt.show()