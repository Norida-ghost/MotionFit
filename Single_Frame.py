import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tkinter import Tk, filedialog

# === STEP 1: Load CSV Data ===
# Load CSV files 
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

if not file_path:
    print("User canceled file selection.")
    exit()

print(f"Loading data from: {file_path}")
data = pd.read_csv(file_path)

n_joints = 34
n_frames = data.shape[0]
joint_data = np.zeros((n_frames, n_joints, 3))

for j in range(n_joints):
    joint_data[:, j, 0] = data[f'X{j}']  # Frontal (X)
    joint_data[:, j, 1] = data[f'Y{j}']  # Vertical (Y)
    joint_data[:, j, 2] = data[f'Z{j}']  # Sagittal (Z)
    
# Joint names (for reference)
Joint_names = [
    'low back', 'left hip', 'right hip', 'Mid back', 'left knee', 'right knee', 'upper back',
    'left ankle', 'right ankle', 'left toe', 'right toe', 'left pinky toe', 'right pinky toe', 'left heel', 'right heel','nose', 'left eye',
    'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist',
    'right wrist', 'left pinky finger', 'right pinky finger', 'left index finger', 'right index finger', 'left thumb', 'right thumb',
    'left hand', 'right hand',
]

label_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 18, 19, 20, 21, 22, 23, 24, 25]

# === STEP 2: Setup Animation ===
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter([], [], [], s=60, c='k')

# Define lines for body parts
lines = {
    'left_leg': ax.plot([], [], [], 'b-', lw=2)[0],
    'right_leg': ax.plot([], [], [], 'r-', lw=2)[0],
    'left_thigh': ax.plot([], [], [], 'b-', lw=2)[0],
    'right_thigh': ax.plot([], [], [], 'r-', lw=2)[0],
    'left_pelvis': ax.plot([], [], [], 'b-', lw=2)[0],
    'right_pelvis': ax.plot([], [], [], 'r-', lw=2)[0],
    'spine': ax.plot([], [], [], 'k-', lw=2)[0],
    'left arm': ax.plot([], [], [], 'b-', lw=2)[0],
    'right arm': ax.plot([], [], [], 'r-', lw=2)[0],
    'left shoulder': ax.plot([], [], [], 'b-', lw=2)[0],
    'right shoulder': ax.plot([], [], [], 'r-', lw=2)[0],
    'right foot toe': ax.plot([], [], [], 'r-', lw=2)[0],
    'right pinky toe': ax.plot([], [], [], 'r-', lw=2)[0],
    'right foot heel': ax.plot([], [], [], 'r-', lw=2)[0],
    'left foot toe': ax.plot([], [], [], 'b-', lw=2)[0],
    'left pinky toe': ax.plot([], [], [], 'b-', lw=2)[0],
    'left foot heel': ax.plot([], [], [], 'b-', lw=2)[0],
    'left hand': ax.plot([], [], [], 'b-', lw=2)[0],
    'right hand': ax.plot([], [], [], 'r-', lw=2)[0],
    'left wrist': ax.plot([], [], [], 'b-', lw=2)[0],
    'right wrist': ax.plot([], [], [], 'r-', lw=2)[0],
    'left thumb': ax.plot([], [], [], 'b-', lw=2)[0],
    'right thumb': ax.plot([], [], [], 'r-', lw=2)[0],
    'left index': ax.plot([], [], [], 'b-', lw=2)[0],
    'right index': ax.plot([], [], [], 'r-', lw=2)[0],
    'left pinky finger': ax.plot([], [], [], 'b-', lw=2)[0],
    'right pinky finger': ax.plot([], [], [], 'r-', lw=2)[0],  
    'head nose': ax.plot([], [], [], 'k-', lw=2)[0],
    'head left eye': ax.plot([], [], [], 'k-', lw=2)[0],
    'head right eye': ax.plot([], [], [], 'k-', lw=2)[0],
    'head left ear': ax.plot([], [], [], 'k-', lw=2)[0],
    'head right ear': ax.plot([], [], [], 'k-', lw=2)[0],
}

# Axis and view settings
ax.set_xlabel('X (Frontal)')
ax.set_ylabel('Y (Vertical)')
ax.set_zlabel('Z (Sagittal)')
ax.set_title('3D Joint Animation')

ax.set_xlim(np.min(joint_data[:, :, 0]), np.max(joint_data[:, :, 0]))
ax.set_ylim(np.min(joint_data[:, :, 1]), np.max(joint_data[:, :, 1]))
ax.set_zlim(np.min(joint_data[:, :, 2]), np.max(joint_data[:, :, 2]))

plt.ion()  # Turn on interactive mode
step = 2

joint_labels = {}
for i in label_indices:
    joint_labels[i] = ax.text(0, 0, 0, Joint_names[i], fontsize=8, color='k')

# === STEP 3: Run Animation ===

for frame_idx in range(0, n_frames, step):
    x = joint_data[frame_idx, :, 0]
    y = joint_data[frame_idx, :, 1]
    z = joint_data[frame_idx, :, 2]
    scat._offsets3d = (x, y, z)

    # Update a few skeleton segments (you can add more)
    lines['left_leg'].set_data([x[7], x[4]], [y[7], y[4]])
    lines['left_leg'].set_3d_properties([z[7], z[4]])
    lines['right_leg'].set_data([x[8], x[5]], [y[8], y[5]])
    lines['right_leg'].set_3d_properties([z[8], z[5]])
    lines['left_thigh'].set_data([x[1], x[4]], [y[1], y[4]])
    lines['left_thigh'].set_3d_properties([z[1], z[4]])
    lines['right_thigh'].set_data([x[2], x[5]], [y[2], y[5]])
    lines['right_thigh'].set_3d_properties([z[2], z[5]])
    lines['left_pelvis'].set_data([x[1], x[0]], [y[1], y[0]])
    lines['left_pelvis'].set_3d_properties([z[1], z[0]])
    lines['right_pelvis'].set_data([x[2], x[0]], [y[2], y[0]])
    lines['right_pelvis'].set_3d_properties([z[2], z[0]])
    lines['spine'].set_data([x[0], x[3], x[6]], [y[0], y[3], y[6]])
    lines['spine'].set_3d_properties([z[0], z[3], z[6]])
    lines['left arm'].set_data([x[20], x[22], x[24]], [y[20], y[22], y[24]])
    lines['left arm'].set_3d_properties([z[20], z[22], z[24]])
    lines['right arm'].set_data([x[21], x[23], x[25]], [y[21], y[23], y[25]])
    lines['right arm'].set_3d_properties([z[21], z[23], z[25]])
    lines['left shoulder'].set_data([x[6], x[20]], [y[6], y[20]])
    lines['left shoulder'].set_3d_properties([z[6], z[20]])
    lines['right shoulder'].set_data([x[6], x[21]], [y[6], y[21]])
    lines['right shoulder'].set_3d_properties([z[6], z[21]])
    lines['right foot toe'].set_data([x[8], x[10]], [y[8], y[10]])
    lines['right foot toe'].set_3d_properties([z[8], z[10]])
    lines['right pinky toe'].set_data([x[8], x[12]], [y[8], y[12]])
    lines['right pinky toe'].set_3d_properties([z[8], z[12]])
    lines['right foot heel'].set_data([x[8], x[14]], [y[8], y[14]])
    lines['right foot heel'].set_3d_properties([z[8], z[14]])
    lines['left foot toe'].set_data([x[7], x[9]], [y[7], y[9]])
    lines['left foot toe'].set_3d_properties([z[7], z[9]])
    lines['left pinky toe'].set_data([x[7], x[11]], [y[7], y[11]])
    lines['left pinky toe'].set_3d_properties([z[7], z[11]])
    lines['left foot heel'].set_data([x[7], x[13]], [y[7], y[13]])
    lines['left foot heel'].set_3d_properties([z[7], z[13]])
    lines['left hand'].set_data([x[24], x[32]], [y[24], y[32]])
    lines['left hand'].set_3d_properties([z[24], z[32]])
    lines['right hand'].set_data([x[25], x[33]], [y[25], y[33]])
    lines['right hand'].set_3d_properties([z[25], z[33]])
    lines['left wrist'].set_data([x[22], x[24]], [y[22], y[24]])
    lines['left wrist'].set_3d_properties([z[22], z[24]])
    lines['right wrist'].set_data([x[23], x[25]], [y[23], y[25]])
    lines['right wrist'].set_3d_properties([z[23], z[25]])
    lines['left thumb'].set_data([x[24], x[30]], [y[24], y[30]])
    lines['left thumb'].set_3d_properties([z[24], z[30]])
    lines['right thumb'].set_data([x[25], x[31]], [y[25], y[31]])
    lines['right thumb'].set_3d_properties([z[25], z[31]])
    lines['left index'].set_data([x[24], x[28]], [y[24], y[28]])
    lines['left index'].set_3d_properties([z[24], z[28]])
    lines['right index'].set_data([x[25], x[29]], [y[25], y[29]])
    lines['right index'].set_3d_properties([z[25], z[29]])
    lines['left pinky finger'].set_data([x[24], x[26]], [y[24], y[26]])
    lines['left pinky finger'].set_3d_properties([z[26], z[26]])
    lines['right pinky finger'].set_data([x[25], x[27]], [y[25], y[27]])
    lines['right pinky finger'].set_3d_properties([z[25], z[27]])
    lines['head nose'].set_data([x[15], x[16]], [y[15], y[16]])
    lines['head nose'].set_3d_properties([z[15], z[16]])
    lines['head left eye'].set_data([x[16], x[18]], [y[16], y[18]])
    lines['head left eye'].set_3d_properties([z[16], z[18]])
    lines['head right eye'].set_data([x[15], x[17]], [y[15], y[17]])
    lines['head right eye'].set_3d_properties([z[15], z[17]])
    lines['head left ear'].set_data([x[15], x[18]], [y[15], y[18]])
    lines['head left ear'].set_3d_properties([z[15], z[18]])
    lines['head right ear'].set_data([x[15], x[19]], [y[15], y[19]])
    lines['head right ear'].set_3d_properties([z[15], z[19]])
    
for i in label_indices:
    joint_labels[i].set_position((x[i], y[i]))
    joint_labels[i].set_3d_properties(z[i])
    joint_color = ['black'] * n_joints
    for i in [1, 4, 7, 19, 21, 23, 25, 26, 28, 30, 32]:
        joint_color[i] = 'blue'
    for i in [2, 5, 8, 20, 22, 24, 25, 27, 29, 31, 33]:
        joint_color[i] = 'red'
    joint_labels[i] = ax.text(0, 0, 0, Joint_names[i], fontsize=8, color=joint_color[i])      

# Axis and view settings
ax.set_xlabel('X (Frontal)')
ax.set_ylabel('Y (Vertical)')
ax.set_zlabel('Z (Sagittal)')
ax.set_title('3D Joint Animation')

ax.set_xlim(np.min(joint_data[:, :, 0]), np.max(joint_data[:, :, 0]))
ax.set_ylim(np.min(joint_data[:, :, 1]), np.max(joint_data[:, :, 1]))
ax.set_zlim(np.min(joint_data[:, :, 2]), np.max(joint_data[:, :, 2]))

plt.ion()  # Turn on interactive mode
step = 2

joint_labels = {}
for i in label_indices:
    joint_labels[i] = ax.text(0, 0, 0, Joint_names[i], fontsize=8, color='k')

# === STEP 3: Run Animation ===
for frame_idx in range(0, n_frames, step):
    x = joint_data[frame_idx, :, 0]
    y = joint_data[frame_idx, :, 1]
    z = joint_data[frame_idx, :, 2]
    scat._offsets3d = (x, y, z)
    
# Update skeleton segments
    for key, line in lines.items():
        # Example: Update each line based on its corresponding joints
        pass  # Add specific updates for each line here

    for i in label_indices:
        joint_labels[i].set_position((x[i], y[i]))
        joint_labels[i].set_3d_properties(z[i])
        
plt.draw()
plt.pause(0.01)

plt.ioff()
plt.show()