import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from tkinter import Tk, filedialog

def load_joint_data(file_path, n_joints=34):
    """
    Load joint data from a CSV file and return a 3D array of joint positions.
    """
    print(f"Loading data from: {file_path}")
    data = pd.read_csv(file_path)
    n_frames = len(data)
    joint_data = np.zeros((n_frames, n_joints, 3))
    for j in range(n_joints):
        joint_data[:, j, 0] = data[f"X{j}"]
        joint_data[:, j, 1] = data[f"Y{j}"]
        joint_data[:, j, 2] = data[f"Z{j}"]
    return joint_data

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

# ------------------------ Setup Plot ------------------------
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Joint Animation with Connections")

colors = ['blue', 'red']
datasets = [joint_data1, joint_data2]

scatters = [
    ax.scatter([], [], [], s=60, c=color, label=f'Dataset {i+1}')
    for i, color in enumerate(colors)
]

# Define connections (pairs of joint indices)
connections = [
    (7, 4), (8, 5), (1, 4), (2, 5), (1, 0), (2, 0), (0, 3), (3, 6),
    (20, 22), (22, 24), (21, 23), (23, 25), (6, 20), (6, 21),
    (8,10), (8,12), (8,14), (7,9), (7,11), (7,13),
    (24,32), (25,33), (22,24), (23,25),
    (24,30), (25,31), (24,28), (25,29), (24,26), (25,27),
    (15,16), (16,18), (15,17), (15,18), (15,19)
]

#^^^Every connection is a line between two joints^^^ Below is the list of connections

#Left ankle -> Left knee (7, 4)
#Right ankle -> Right knee (8, 5)
#Left hip -> Left knee (1, 4)
#Right hip -> Right knee (2, 5)
# Left hip -> Low back (1, 0)
# Right hip -> Low back (2, 0)
# Low back -> Mid back (0, 3)
# Mid back -> Upper back (3, 6)
# Left shoulder -> Left elbow (20, 22)
# Left elbow -> Left wrist (22, 24)
# Right shoulder -> Right elbow (21, 23)
# Right elbow -> Right wrist (23, 25)
# Upper back -> Left shoulder (6, 20)
# Upper back -> Right shoulder (6, 21)
# Right ankle -> Right toe (8, 10)
# Right ankle -> Right pinky toe (8, 12)
# Right ankle -> Right heel (8, 14)
# Left ankle -> Left toe (7, 9)
# Left ankle -> Left pinky toe (7, 11)
# Left ankle -> Left heel (7, 13)
# Left wrist -> Left pinky finger (24, 32)
# Right wrist -> Right pinky finger (25, 33)
# Left elbow -> Left wrist (22, 24)
# Right elbow -> Right wrist (23, 25)
# Left wrist -> Left index finger (24, 30)
# Right wrist -> Right index finger (25, 31)
# Left wrist -> Left thumb (24, 28)
# Right wrist -> Right thumb (25, 29)
# Left pinky finger -> Left pinky finger (24, 26)
# Right pinky finger -> Right pinky finger (25, 27)
# Nose -> Left eye (15, 16)
# Nose -> Right eye (15, 17)
# Left eye -> Left ear (16, 18)
# Right eye -> Right ear (17, 19)

# Create lines for each dataset
lines = []
for i in range(2):  # one for each dataset
    line_set = []
    for conn in connections:
        color = colors[i]
        line, = ax.plot([], [], [], color=color, lw=2)
        line_set.append(line)
    lines.append(line_set)

# ------------------------ Joint Labels ------------------------
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
    labels.append([
        ax.text(0, 0, 0, '', fontsize=8, color=color)
        for _ in label_indices
    ])

# ------------------------ Set Axes Limits ------------------------
combined = np.concatenate((joint_data1, joint_data2), axis=0)
ax.set_xlim(np.min(combined[:, :, 0]), np.max(combined[:, :, 0]))
ax.set_ylim(np.min(combined[:, :, 1]), np.max(combined[:, :, 1]))
ax.set_zlim(np.min(combined[:, :, 2]), np.max(combined[:, :, 2]))

ax.set_xlabel('X (Frontal)')
ax.set_ylabel('Y (Vertical)')
ax.set_zlabel('Z (Sagittal)')
ax.legend()

# ------------------------ Animation Function ------------------------
def update(frame):
    for d in range(2):  # dataset 0 and 1
        data = datasets[d]
        x = data[frame, :, 0]
        y = data[frame, :, 1]
        z = data[frame, :, 2]

        # Update joints
        scatters[d]._offsets3d = (x, y, z)

        # Update lines
        for l, (i, j) in zip(lines[d], connections):
            l.set_data([x[i], x[j]], [y[i], y[j]])
            l.set_3d_properties([z[i], z[j]])

        # Update labels
        for idx, joint_idx in enumerate(label_indices):
            labels[d][idx].set_position((x[joint_idx], y[joint_idx]))
            labels[d][idx].set_3d_properties(z[joint_idx])
            labels[d][idx].set_text(Joint_names[joint_idx])

    return scatters + sum(lines, []) + sum(labels, [])

# ------------------------ Run Animation ------------------------
step = 2
ani = FuncAnimation(fig, update, frames=range(0, n_frames, step), interval=30, blit=False)
plt.show()
