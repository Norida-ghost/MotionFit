import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def update_3d_plot(frame, joint_data1, joint_data2, scatter1, scatter2, text1, text2, label_indices):
    """
    Update the 3D scatter plot for the given frame.
    """
    # Update first dataset
    x1 = joint_data1[frame, label_indices, 0]
    y1 = joint_data1[frame, label_indices, 1]
    z1 = joint_data1[frame, label_indices, 2]
    scatter1._offsets3d = (x1, y1, z1)
    for i, txt in enumerate(label_indices):
        text1[i].set_position((x1[i], y1[i]))
        text1[i].set_3d_properties(z1[i])
        text1[i].set_text(f"J{txt}")

    # Update second dataset
    x2 = joint_data2[frame, label_indices, 0]
    y2 = joint_data2[frame, label_indices, 1]
    z2 = joint_data2[frame, label_indices, 2]
    scatter2._offsets3d = (x2, y2, z2)
    for i, txt in enumerate(label_indices):
        text2[i].set_position((x2[i], y2[i]))
        text2[i].set_3d_properties(z2[i])
        text2[i].set_text(f"J{txt}")

# File selection
Tk().withdraw()
print("Select the first CSV file.")
file_path1 = filedialog.askopenfilename(title="Select the first CSV file", filetypes=[("CSV files", "*.csv")])
if not file_path1:
    print("First file not selected. Exiting.")
    exit()

print("Select the second CSV file.")
file_path2 = filedialog.askopenfilename(title="Select the second CSV file", filetypes=[("CSV files", "*.csv")])
if not file_path2:
    print("Second file not selected. Exiting.")
    exit()

# Load joint data
n_joints = 34
joint_data1 = load_joint_data(file_path1, n_joints)
joint_data2 = load_joint_data(file_path2, n_joints)

n_frames = min(len(joint_data1), len(joint_data2))  # Use the minimum number of frames for synchronization
label_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Indices of joints to display

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plots for both datasets
scatter1 = ax.scatter([], [], [], s=60, c='blue', label='File 1')
scatter2 = ax.scatter([], [], [], s=60, c='red', label='File 2')

# Text labels for joints
text1 = [ax.text(0, 0, 0, '', fontsize=9, weight='bold', color='blue') for _ in label_indices]
text2 = [ax.text(0, 0, 0, '', fontsize=9, weight='bold', color='red') for _ in label_indices]

# Set plot limits
all_coords = np.concatenate((joint_data1, joint_data2), axis=0)
ax.set_xlim(np.min(all_coords[:, :, 0]), np.max(all_coords[:, :, 0]))
ax.set_ylim(np.min(all_coords[:, :, 1]), np.max(all_coords[:, :, 1]))
ax.set_zlim(np.min(all_coords[:, :, 2]), np.max(all_coords[:, :, 2]))

ax.set_xlabel('X (Frontal)')
ax.set_ylabel('Y (Vertical)')
ax.set_zlabel('Z (Sagittal)')
ax.set_title('3D Joint Positions Comparison')
ax.legend()

# Animate the frames
for frame in range(0, n_frames, 2):  # Step by 2 for smoother animation
    update_3d_plot(frame, joint_data1, joint_data2, scatter1, scatter2, text1, text2, label_indices)
    plt.draw()
    plt.pause(0.03)

plt.show()