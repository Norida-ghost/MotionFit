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

file1_path = filedialog.askopenfilename(title="Select the 8-camera CSV file (Gold Standard)",
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
labels = ['8-camera (Gold Standard)', 'Compared File']

# === QUALITY COMPARISON METRICS ===

def calculate_euclidean_distance(data1, data2):
    """Calculate 3D Euclidean distance between corresponding joints"""
    return np.sqrt(np.sum((data1 - data2)**2, axis=2))

def calculate_rmse(data1, data2):
    """Calculate Root Mean Square Error"""
    diff = data1 - data2
    mse = np.mean(diff**2)
    return np.sqrt(mse)

def calculate_mae(data1, data2):
    """Calculate Mean Absolute Error"""
    diff = np.abs(data1 - data2)
    return np.mean(diff)

def calculate_jitter(data):
    """Calculate jitter (smoothness) - higher values indicate more noise"""
    velocity = np.diff(data, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jitter = np.mean(np.sqrt(np.sum(acceleration**2, axis=2)))
    return jitter

def calculate_correlation(data1, data2):
    """Calculate correlation coefficient between trajectories"""
    correlations = []
    for joint in range(n_joints):
        for axis in range(3):
            corr = np.corrcoef(data1[:, joint, axis], data2[:, joint, axis])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    return np.mean(correlations)

# === Perform Quality Analysis ===
print("\n" + "="*50)
print("MOTION CAPTURE QUALITY ANALYSIS")
print("="*50)

# 1. Overall RMSE (Root Mean Square Error)
overall_rmse = calculate_rmse(joint_data1, joint_data2)
print(f"Overall RMSE: {overall_rmse:.4f} mm")

# 2. Overall MAE (Mean Absolute Error)
overall_mae = calculate_mae(joint_data1, joint_data2)
print(f"Overall MAE: {overall_mae:.4f} mm")

# 3. Per-joint analysis
euclidean_distances = calculate_euclidean_distance(joint_data1, joint_data2)
mean_distances_per_joint = np.mean(euclidean_distances, axis=0)
std_distances_per_joint = np.std(euclidean_distances, axis=0)

print(f"\nPer-Joint Analysis:")
print(f"{'Joint':<20} {'Mean Error (mm)':<15} {'Std Error (mm)':<15}")
print("-" * 50)

worst_joints = []
for i in range(n_joints):
    error = mean_distances_per_joint[i]
    std_error = std_distances_per_joint[i]
    print(f"{joint_names[i]:<20} {error:<15.3f} {std_error:<15.3f}")
    if error > np.percentile(mean_distances_per_joint, 75):  # Top 25% worst
        worst_joints.append((joint_names[i], error))

print(f"\nWorst performing joints (top 25%):")
for joint, error in sorted(worst_joints, key=lambda x: x[1], reverse=True):
    print(f"  {joint}: {error:.3f} mm")

# 4. Temporal analysis 
mean_error_per_frame = np.mean(euclidean_distances, axis=1)
max_error_frame = np.argmax(mean_error_per_frame)
min_error_frame = np.argmin(mean_error_per_frame)

print(f"\nTemporal Analysis:")
print(f"Average error per frame: {np.mean(mean_error_per_frame):.3f} mm")
print(f"Maximum error frame: {max_error_frame} (error: {mean_error_per_frame[max_error_frame]:.3f} mm)")
print(f"Minimum error frame: {min_error_frame} (error: {mean_error_per_frame[min_error_frame]:.3f} mm)")

# 5. Smoothness comparison (jitter)
jitter_gold = calculate_jitter(joint_data1)
jitter_comparison = calculate_jitter(joint_data2)
jitter_ratio = jitter_comparison / jitter_gold

print(f"\nSmoothness Analysis:")
print(f"Gold standard jitter: {jitter_gold:.4f}")
print(f"Comparison file jitter: {jitter_comparison:.4f}")
print(f"Jitter ratio: {jitter_ratio:.4f} ({'worse' if jitter_ratio > 1 else 'better'} smoothness)")

# 6. Correlation analysis
correlation = calculate_correlation(joint_data1, joint_data2)
print(f"\nCorrelation Analysis:")
print(f"Average correlation: {correlation:.4f} (1.0 = perfect, 0.0 = no correlation)")

# 7. Quality percentage
# Normalize errors to percentage (assuming 1mm error = 1% quality loss)
max_acceptable_error = 5.0  # mm - you can adjust this threshold
quality_percentage = max(0, (1 - overall_rmse / max_acceptable_error) * 100)
print(f"\nOverall Quality Assessment:")
print(f"Quality retention: {quality_percentage:.1f}%")
print(f"Quality loss: {100 - quality_percentage:.1f}%")

# === VISUALIZATION ===

# Ask user what visualizations they want
visualizations = messagebox.askyesno("Visualizations", "Do you want to see detailed comparison plots?")

if visualizations:
    # 1. Error heatmap per joint over time
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Motion Capture Quality Analysis', fontsize=16)
    
    # Heatmap of errors
    ax1 = axes[0, 0]
    im = ax1.imshow(euclidean_distances[:100, :].T, aspect='auto', cmap='viridis')
    ax1.set_title('Error Heatmap (First 100 frames)')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Joint')
    ax1.set_yticks(range(0, n_joints, 5))
    ax1.set_yticklabels([joint_names[i] for i in range(0, n_joints, 5)], fontsize=8)
    plt.colorbar(im, ax=ax1, label='Error (mm)')
    
    # Bar chart of per-joint errors
    ax2 = axes[0, 1]
    bars = ax2.bar(range(n_joints), mean_distances_per_joint, color='skyblue', alpha=0.7)
    ax2.set_title('Average Error per Joint')
    ax2.set_xlabel('Joint Index')
    ax2.set_ylabel('Average Error (mm)')
    ax2.set_xticks(range(0, n_joints, 5))
    
    # Highlight worst joints
    for i, bar in enumerate(bars):
        if mean_distances_per_joint[i] > np.percentile(mean_distances_per_joint, 75):
            bar.set_color('red')
            bar.set_alpha(0.8)
    
    # Error over time
    ax3 = axes[1, 0]
    ax3.plot(mean_error_per_frame, color='red', linewidth=1)
    ax3.set_title('Error Over Time')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Average Error (mm)')
    ax3.axhline(y=np.mean(mean_error_per_frame), color='blue', linestyle='--', 
                label=f'Mean: {np.mean(mean_error_per_frame):.3f} mm')
    ax3.legend()
    
    # Distribution of errors
    ax4 = axes[1, 1]
    ax4.hist(euclidean_distances.flatten(), bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
    ax4.set_title('Error Distribution')
    ax4.set_xlabel('Error (mm)')
    ax4.set_ylabel('Frequency')
    ax4.axvline(x=np.mean(euclidean_distances), color='blue', linestyle='--', 
                label=f'Mean: {np.mean(euclidean_distances):.3f} mm')
    ax4.legend()
    
    plt.tight_layout()
    plt.show(block=False)

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
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X (Frontal)')
ax.set_ylabel('Y (Vertical)')
ax.set_zlabel('Z (Sagittal)')
ax.set_title(f'Static Comparison - Overall RMSE: {overall_rmse:.3f} mm')
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

# === Ask for animation ===
show_anim = messagebox.askyesno("Show Animation?", "Do you want to view the 3D joint animation?")

if show_anim:
    fig_anim = plt.figure(figsize=(12, 8))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    ax_anim.set_xlabel('X (Frontal)')
    ax_anim.set_ylabel('Y (Vertical)')
    ax_anim.set_zlabel('Z (Sagittal)')
    ax_anim.set_title(f'3D Joint Animation - Quality: {quality_percentage:.1f}%')
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

# === Save results to file ===
save_results = messagebox.askyesno("Save Results", "Do you want to save the analysis results to a file?")

if save_results:
    results_file = filedialog.asksaveasfilename(
        title="Save Analysis Results",
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    
    if results_file:
        with open(results_file, 'w') as f:
            f.write("MOTION CAPTURE QUALITY ANALYSIS RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Gold Standard File: {file1_path}\n")
            f.write(f"Comparison File: {file2_path}\n")
            f.write(f"Number of frames analyzed: {n_frames}\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write(f"Overall RMSE: {overall_rmse:.4f} mm\n")
            f.write(f"Overall MAE: {overall_mae:.4f} mm\n")
            f.write(f"Correlation: {correlation:.4f}\n")
            f.write(f"Quality retention: {quality_percentage:.1f}%\n")
            f.write(f"Quality loss: {100 - quality_percentage:.1f}%\n\n")
            
            f.write("SMOOTHNESS ANALYSIS:\n")
            f.write(f"Gold standard jitter: {jitter_gold:.4f}\n")
            f.write(f"Comparison file jitter: {jitter_comparison:.4f}\n")
            f.write(f"Jitter ratio: {jitter_ratio:.4f} ({'worse' if jitter_ratio > 1 else 'better'} smoothness)\n\n")
            
            f.write("WORST PERFORMING JOINTS:\n")
            for joint, error in sorted(worst_joints, key=lambda x: x[1], reverse=True):
                f.write(f"  {joint}: {error:.3f} mm\n")
                
        print(f"Results saved to: {results_file}")

print("\nAnalysis complete! Check the visualizations and saved results.")
input("Press Enter to exit...")
