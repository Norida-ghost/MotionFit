import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from tkinter import Tk, filedialog
import os

# ------------------------ Load Joint Data ------------------------
def load_joint_data(file_path, n_joints=34):
    print(f"Loading data from: {file_path}")
    data = pd.read_csv(file_path)
    n_frames = len(data)
    joint_data = np.zeros((n_frames, n_joints, 3))
    for j in range(n_joints):
        joint_data[:, j, 0] = data[f"X{j}"]
        joint_data[:, j, 1] = data[f"Y{j}"]
        joint_data[:, j, 2] = data[f"Z{j}"]
    return joint_data

# ------------------------ Analysis Functions ------------------------
def detect_exercise(joint_data):
    left_knee_y = joint_data[:, 4, 1]
    right_knee_y = joint_data[:, 5, 1]

    movement_std = np.std(left_knee_y - right_knee_y)
    knee_range = np.max(left_knee_y) - np.min(left_knee_y)

    if movement_std > 20:
        return "Jumping / Running"
    elif knee_range > 10:
        return "Squatting / Knee Bend"
    else:
        return "Static or Unknown"

def detect_dominant_side(joint_data):
    left_indices = [1, 4, 7, 20, 22, 24]
    right_indices = [2, 5, 8, 21, 23, 25]

    left_movement = np.sum(np.linalg.norm(np.diff(joint_data[:, left_indices, :], axis=0), axis=2))
    right_movement = np.sum(np.linalg.norm(np.diff(joint_data[:, right_indices, :], axis=0), axis=2))

    if left_movement > right_movement:
        return "Left side dominant"
    elif right_movement > left_movement:
        return "Right side dominant"
    else:
        return "Symmetrical"

def detect_global_direction(joint_data):
    center_trajectory = np.mean(joint_data, axis=1)
    delta = center_trajectory[-1] - center_trajectory[0]

    directions = []
    if abs(delta[0]) > 10:
        directions.append("Right" if delta[0] > 0 else "Left")
    if abs(delta[1]) > 10:
        directions.append("Up" if delta[1] > 0 else "Down")
    if abs(delta[2]) > 10:
        directions.append("Forward" if delta[2] > 0 else "Backward")

    return ", ".join(directions) if directions else "No significant movement"

# ------------------------ Select Files ------------------------
Tk().withdraw()
print("Select the first CSV file.")
file1 = filedialog.askopenfilename(title="Select the first CSV file", filetypes=[("CSV files", "*.csv")])
if not file1:
    print("No file selected. Exiting.")
    exit()

print("Select the second CSV file.")
file2 = filedialog.askopenfilename(title="Select the second CSV file", filetypes=[("CSV files", "*.csv")])
if not file2:
    print("Second file not selected. Exiting.")
    exit()

# ------------------------ Load Data ------------------------
n_joints = 34
joint_data1 = load_joint_data(file1, n_joints)
joint_data2 = load_joint_data(file2, n_joints)
n_frames = min(joint_data1.shape[0], joint_data2.shape[0])
datasets = [joint_data1, joint_data2]

# ------------------------ Perform Analysis ------------------------
print("\n--- Analysis Results ---")
print(f"File 1: {os.path.basename(file1)}")
print("  Exercise Type:      ", detect_exercise(joint_data1))
print("  Dominant Side:      ", detect_dominant_side(joint_data1))
print("  Global Movement:    ", detect_global_direction(joint_data1))

print(f"\nFile 2: {os.path.basename(file2)}")
print("  Exercise Type:      ", detect_exercise(joint_data2))
print("  Dominant Side:      ", detect_dominant_side(joint_data2))
print("  Global Movement:    ", detect_global_direction(joint_data2))
print("------------------------\n")

# ------------------------ Setup Plot ------------------------
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Joint Animation with Connections")

colors = ['blue', 'red']
file1_name = os.path.basename(file1)
file2_name = os.path.basename(file2)

scatters = [
    ax.scatter([], [], [], s=60, c=color, label=file_name)
    for color, file_name in zip(colors, [file1_name, file2_name])
]
ax.legend()

connections = [
    (7, 4), (8, 5), (1, 4), (2, 5), (1, 0), (2, 0), (0, 3), (3, 6),
    (20, 22), (22, 24), (21, 23), (23, 25), (6, 20), (6, 21),
    (8,10), (8,12), (8,14), (7,9), (7,11), (7,13),
    (24,32), (25,33), (22,24), (23,25),
    (24,30), (25,31), (24,28), (25,29), (24,26), (25,27),
    (15,16), (16,18), (15,17), (15,18), (15,19)
]

lines = []
for i in range(2):
    line_set = []
    for conn in connections:
        color = colors[i]
        line, = ax.plot([], [], [], color=color, lw=2)
        line_set.append(line)
    lines.append(line_set)

Joint_names = [
    'low back', 'left hip', 'right hip', 'Mid back', 'left knee', 'right knee', 'upper back',
    'left ankle', 'right ankle', 'left toe', 'right toe', 'left pinky toe', 'right pinky toe',
    'left heel', 'right heel', 'nose', 'left eye', 'right eye', 'left ear', 'right ear',
    'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist',
    'left pinky finger', 'right pinky finger', 'left index finger', 'right index finger',
    'left thumb', 'right thumb', 'left hand', 'right hand',
]
label_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 18, 19, 20, 21, 22, 23, 24, 25]

labels = []
for i, color in enumerate(colors):
    labels.append([ax.text(0, 0, 0, '', fontsize=8, color=color) for _ in label_indices])

combined = np.concatenate((joint_data1, joint_data2), axis=0)
ax.set_xlim(np.min(combined[:, :, 0]), np.max(combined[:, :, 0]))
ax.set_ylim(np.min(combined[:, :, 1]), np.max(combined[:, :, 1]))
ax.set_zlim(np.min(combined[:, :, 2]), np.max(combined[:, :, 2]))
ax.set_xlabel('X (Frontal)')
ax.set_ylabel('Y (Vertical)')
ax.set_zlabel('Z (Sagittal)')

# ------------------------ Animation ------------------------
def update(frame):
    for d in range(2):
        data = datasets[d]
        x = data[frame, :, 0]
        y = data[frame, :, 1]
        z = data[frame, :, 2]

        scatters[d]._offsets3d = (x, y, z)

        for l, (i, j) in zip(lines[d], connections):
            l.set_data([x[i], x[j]], [y[i], y[j]])
            l.set_3d_properties([z[i], z[j]])

        if d == 0:
            for idx, joint_idx in enumerate(label_indices):
                labels[d][idx].set_position((x[joint_idx], y[joint_idx]))
                labels[d][idx].set_3d_properties(z[joint_idx])
                labels[d][idx].set_text(Joint_names[joint_idx])
        else:
            for label in labels[d]:
                label.set_text('')
    return scatters + sum(lines, []) + sum(labels, [])

step = 2
ani = FuncAnimation(fig, update, frames=range(0, n_frames, step), interval=30, blit=False)
plt.show()
