import os
import sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# === Settings ===
n_joints = 34
frame_rate = 30

joint_names = [
    'Low back', 'Left Hip', 'Right Hip', 'Middle back', 'Left Knee', 'Right Knee',
    'Upper back', 'Left Ankle', 'Right Ankle', 'Left Toe', 'Right Toe',
    'Left 5th Toe', 'Right 5th Toe', 'Left Calcaneus', 'Right Calcaneus',
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left 5th Finger', 'Right 5th Finger',
    'Left 3rd Finger', 'Right 3rd Finger', 'Left Thumb', 'Right Thumb',
    'Left carpus', 'Right carpus'
]

# === GUI to select files ===
root = tk.Tk()
root.withdraw()  # Hide the main window

default_folder = r'C:\Users\edm\OneDrive - Aalborg Universitet\ESA - E4D project\MoCap_Supporting_Files'
if not os.path.exists(default_folder):
    messagebox.showwarning("Warning", "Default folder not found. Using current directory.")
    default_folder = os.getcwd()

file1_path = filedialog.askopenfilename(title="Select the 8-camera CSV file",
                                        initialdir=default_folder,
                                        filetypes=[("CSV files", "*.csv")])
if not file1_path:
    print("User cancelled gold standard file selection. Exiting.")
    sys.exit()

file2_path = filedialog.askopenfilename(title="Select the FILE TO COMPARE",
                                        initialdir=default_folder,
                                        filetypes=[("CSV files", "*.csv")])
if not file2_path:
    print("User cancelled comparison file selection. Exiting.")
    sys.exit()

print(f"Gold Standard loaded: {file1_path}")
print(f"Comparison File loaded: {file2_path}")

# === Load data ===
data1 = pd.read_csv(file1_path)
data2 = pd.read_csv(file2_path)

n_frames = min(len(data1), len(data2))

joint_data1 = np.zeros((n_frames, n_joints, 3))
joint_data2 = np.zeros((n_frames, n_joints, 3))

for j in range(n_joints):
    joint_data1[:, j, 0] = data1[f'X{j}'][:n_frames]
    joint_data1[:, j, 1] = data1[f'Y{j}'][:n_frames]
    joint_data1[:, j, 2] = data1[f'Z{j}'][:n_frames]

    joint_data2[:, j, 0] = data2[f'X{j}'][:n_frames]
    joint_data2[:, j, 1] = data2[f'Y{j}'][:n_frames]
    joint_data2[:, j, 2] = data2[f'Z{j}'][:n_frames]

datasets = [joint_data1, joint_data2]
colors = ['k', 'r']
labels = ['8-camera', 'Compared File']

# === Skeleton connections ===
skeleton_connections = [
    (8, 5, 2), (9, 6, 3),
    (1, 2), (1, 3),
    (1, 4, 7),
    (21, 23, 25), (22, 24, 26),
    (7, 21), (7, 22),
    (25, 27), (25, 29), (25, 31), (25, 33),
    (26, 28), (26, 30), (26, 32), (26, 34),
    (7, 16), (16, 17), (16, 18), (17, 19), (18, 20),
    (16, 19), (16, 20), (19, 20),
    (8, 10), (8, 12), (8, 14),
    (9, 11), (9, 13), (9, 15)
]

# === Static Plot ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X (Frontal)')
ax.set_ylabel('Y (Vertical)')
ax.set_zlabel('Z (Sagittal)')
ax.set_title('Static plot')
ax.view_init(elev=20, azim=45)

for d, data in enumerate(datasets):
    x = data[0, :, 0]
    y = data[0, :, 1]
    z = data[0, :, 2]
    c = colors[d]

    for conn in skeleton_connections:
        points = np.array([[x[i - 1], y[i - 1], z[i - 1]] for i in conn])
        ax.plot(points[:, 0], points[:, 1], points[:, 2], c=c, lw=2)

    ax.scatter(x, y, z, c=c, s=30, label=labels[d])

ax.legend()
plt.show(block=False)
input("Static frame displayed. Press Enter to continue...")

# === Ask for animation ===
show_anim = messagebox.askyesno("Show Animation?", "Do you want to view the 3D joint animation?")

if show_anim:
    fig_anim = plt.figure(figsize=(10, 8))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    ax_anim.set_xlabel('X (Frontal)')
    ax_anim.set_ylabel('Y (Vertical)')
    ax_anim.set_zlabel('Z (Sagittal)')
    ax_anim.set_title('3D Joint Animation')
    ax_anim.view_init(elev=20, azim=45)

    all_coords = np.concatenate([joint_data1.reshape(-1, 3), joint_data2.reshape(-1, 3)], axis=0)
    mins = all_coords.min(axis=0)
    maxs = all_coords.max(axis=0)
    margin = 0.05
    ranges = maxs - mins
    ax_anim.set_xlim(mins[0] - margin * ranges[0], maxs[0] + margin * ranges[0])
    ax_anim.set_ylim(mins[1] - margin * ranges[1], maxs[1] + margin * ranges[1])
    ax_anim.set_zlim(mins[2] - margin * ranges[2], maxs[2] + margin * ranges[2])

    lines = []
    scatters = []

    for d in range(2):
        lines.append([ax_anim.plot([], [], [], colors[d] + '-', lw=2)[0] for _ in skeleton_connections])
        scatters.append(ax_anim.scatter([], [], [], c=colors[d], s=30))

    def init():
        for d in range(2):
            for line in lines[d]:
                line.set_data([], [])
                line.set_3d_properties([])
            scatters[d]._offsets3d = ([], [], [])
        return sum(lines, []) + scatters

    def update(frame):
        for d, data in enumerate(datasets):
            x = data[frame, :, 0]
            y = data[frame, :, 1]
            z = data[frame, :, 2]

            for idx, conn in enumerate(skeleton_connections):
                points = np.array([[x[i - 1], y[i - 1], z[i - 1]] for i in conn])
                lines[d][idx].set_data(points[:, 0], points[:, 1])
                lines[d][idx].set_3d_properties(points[:, 2])

            scatters[d]._offsets3d = (x, y, z)
        return sum(lines, []) + scatters

    ani = animation.FuncAnimation(fig_anim, update, frames=range(0, n_frames, 6),
                                  init_func=init, blit=False, interval=100)
    plt.show()

