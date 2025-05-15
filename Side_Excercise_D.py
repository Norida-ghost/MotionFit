import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from tkinter import Tk, filedialog
import os

# This script detects the direction of movement in 3D space from joint data
def detect_global_direction(joint_data1, joint_data2):
    center_of_mass_x1 = np.mean(joint_data1[:, :, 0], axis=1)
    center_of_mass_x2 = np.mean(joint_data2[:, :, 0], axis=1)
    movement1 = np.sum(np.diff(center_of_mass_x1))
    movement2 = np.sum(np.diff(center_of_mass_x2))
    total_movement = movement1 + movement2
    if total_movement > 0:
        return "Right"
    elif total_movement < 0:
        return "Left"
    else:
        return "Stationary"
# This function detects the side movement of the subject based on the joint data
def detect_side_movement(joint_data):
    # Calculate the center of mass in the X-axis (lateral movement)
    center_of_mass_x = np.mean(joint_data[:, :, 0], axis=1)
    total_movement_x = np.sum(np.abs(np.diff(center_of_mass_x)))

    # Define a threshold for minimal lateral movement
    lateral_threshold = 0.05  # Adjust this value based on your data

    # Classify side movement
    if total_movement_x < lateral_threshold:
        return "None"  # Minimal lateral movement
    else:
        # Check the direction of movement
        net_movement_x = np.sum(np.diff(center_of_mass_x))
        symmetry_threshold = 0.01  # Close to zero means bilateral
        if abs(net_movement_x) < symmetry_threshold:
            return "Bilateral"
        elif net_movement_x > 0:
            return "Right"
        else:
            return "Left"

# This function detects the direction of animation based on the joint data
def detect_animation_direction(joint_data):
    center_of_mass_x = np.mean(joint_data[:, :, 0], axis=1)
    center_of_mass_y = np.mean(joint_data[:, :, 1], axis=1)
    center_of_mass_z = np.mean(joint_data[:, :, 2], axis=1)

    movement_x = np.sum(np.diff(center_of_mass_x))
    movement_y = np.sum(np.diff(center_of_mass_y))
    movement_z = np.sum(np.diff(center_of_mass_z))

    movements = {'Lateral (X)': abs(movement_x), 'Vertical (Y)': abs(movement_y), 'Horizontal (Z)': abs(movement_z)}
    dominant_direction = max(movements, key=movements.get)

    return dominant_direction
# This function classifies the exercise based on the joint data
def classify_exercise(joint_data):
    # Extract relevant joints
    low_back_y = joint_data[:, 0, 1]
    mid_back_y = joint_data[:, 3, 1]
    upper_back_y = joint_data[:, 6, 1]
    left_hip_y = joint_data[:, 1, 1]
    right_hip_y = joint_data[:, 2, 1]
    left_knee_y = joint_data[:, 4, 1]
    right_knee_y = joint_data[:, 5, 1]
    
    # Lateral (X-axis) movement
    spine_x = np.mean(joint_data[:, [0, 3, 6], 0], axis=1)
    lateral_spine_movement = np.max(spine_x) - np.min(spine_x)

    # Vertical (Y-axis) movement
    hip_y = np.mean(joint_data[:, [1, 2], 1], axis=1)
    knee_y = np.mean(joint_data[:, [4, 5], 1], axis=1)
    back_y = np.mean(joint_data[:, [0, 3, 6], 1], axis=1)
    
    hip_amp = np.max(hip_y) - np.min(hip_y)
    knee_amp = np.max(knee_y) - np.min(knee_y)
    back_amp = np.max(back_y) - np.min(back_y)
    
    # Forward (Z-axis) movement of hips
    hip_z = np.mean(joint_data[:, [1, 2], 2], axis=1)
    hip_forward_movement = np.max(hip_z) - np.min(hip_z)

    # Thresholds (tunable)
    vertical_thresh = 0.15
    lateral_thresh = 0.1
    forward_thresh = 0.1

    # Classification logic
    if lateral_spine_movement > lateral_thresh and hip_amp < vertical_thresh:
        return "Side Flexion", "Lateral"
    elif hip_amp > vertical_thresh and knee_amp > vertical_thresh and hip_forward_movement < forward_thresh:
        return "Squat", "Vertical"
    elif back_amp > vertical_thresh and hip_forward_movement > forward_thresh and knee_amp < vertical_thresh:
        return "Deadlift", "Forward"
    else:
        return "Unknown", "None"
# This function loads joint data from a CSV file
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
# This function plots the joint data in 3D space
Tk().withdraw()
print("Select the first CSV file.")
file1 = filedialog.askopenfilename(title="Select the first CSV file", filetypes=[("CSV files", "*.csv")])
if not file1:
    print("No file selected. Exiting.")
    exit()
## Check if the file exists
print("Select the second CSV file.")
file2 = filedialog.askopenfilename(title="Select the second CSV file", filetypes=[("CSV files", "*.csv")])
if not file2:
    print("Second file not selected. Exiting.")
    exit()
# Setup
n_joints = 34
joint_data1 = load_joint_data(file1, n_joints)
joint_data2 = load_joint_data(file2, n_joints)

n_frames = min(joint_data1.shape[0], joint_data2.shape[0])

exercise1, side1 = classify_exercise(joint_data1)
exercise2, side2 = classify_exercise(joint_data2)

direction1 = detect_animation_direction(joint_data1)
direction2 = detect_animation_direction(joint_data2)

side_movement1 = detect_side_movement(joint_data1)
side_movement2 = detect_side_movement(joint_data2)

print(f"\nDetected Exercise for File 1: {exercise1} | Direction: {direction1} | Side Movement: {side_movement1}")
print(f"Detected Exercise for File 2: {exercise2} | Direction: {direction2} | Side Movement: {side_movement2}\n")
