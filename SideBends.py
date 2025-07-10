from encodings.punycode import T
import os
from tkinter import font
from matplotlib import lines
import numpy as np
import pandas as pd
from pyparsing import line
from scipy.signal import butter, filtfilt
import tkinter as tk
from tkinter import Tk, filedialog, messagebox
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt

#---Choose and load CSV-file---
#Choose main folder
default_folder = r'D:\ESA\ESA - Projekt - May'

if not os.path.exists(default_folder):
    print('Warning: Default folder does not exist. Using current folder instead.')
    default_folder = os.getcwd()

#Use Tkinter to open a file dialog
Tk().withdraw() #Hide the main window
filename = filedialog.askopenfilename(
    initialdir=default_folder,
    title='Select CSV file',
    filetypes=[('CSV files', '*.csv')],
)

if not filename:
    print('No file selected. Exiting.')
    exit()
    
print(f'Loading data from {filename}')
data = pd.read_csv(filename)

print(f"Data shape: {data.shape}")
print(f"Data columns: {data.columns.tolist()}")
# The following print statements are moved after joint_data, frame_idx, x, y, z are defined.

#---Joint-data and room---
n_joints = 34
n_frames = len(data)
joint_data = np.zeros((n_frames, n_joints, 3))

frame_rate = 30 # Hz
time = np.arange(n_frames) / frame_rate

for j in range(n_joints):
    joint_data[:, j, 0] = data[f'X{j}']
    joint_data[:, j, 1] = data[f'Y{j}']
    joint_data[:, j, 2] = data[f'Z{j}']

frame_to_plot = 1

x = joint_data[frame_to_plot, :, 0]  # X coordinates of all joints at this frame
y = joint_data[frame_to_plot, :, 1]  # Y coordinates of all joints at this frame  
z = joint_data[frame_to_plot, :, 2]  # Z coordinates of all joints at this frame
        
#---Butterworth filter---
x = joint_data[frame_to_plot, :, 0]  # X coordinates of all joints at this frame
y = joint_data[frame_to_plot, :, 1]  # Y coordinates of all joints at this frame  
z = joint_data[frame_to_plot, :, 2]  # Z coordinates of all joints at this frame

print(f"Joint data shape: {joint_data.shape}")
print(f"Sample coordinates for frame {frame_to_plot}:")
print(f"X range: {np.min(x)} to {np.max(x)}")
print(f"Y range: {np.min(y)} to {np.max(y)}")
print(f"Z range: {np.min(z)} to {np.max(z)}")
print(f"Any NaN values? {np.isnan(joint_data).any()}")
#Wn = cutoff_freq / nyquist
#b, a = butter(order, Wn, btype='low', output='ba')  # Explicitly set output to 'ba' for (b, a)

#joint_data_filtered = np.zeros_like(joint_data)
#for j in range(n_joints):
    #for dim in range(3):
        #signal = joint_data[:, j, dim]
        #joint_data_filtered[:, j, dim] = filtfilt(b, a, signal)
        
#joint_data = joint_data_filtered

#---Name joints---
joint_names = [
    'Low back', 'Left hip', 'Right hip', 'Middle back', 'Left knee', 'Right knee', 'Upper back', 'Left ankle', 'Right ankle', 'Left toe', 'Right toe', 'Left 5th toe', 'Right 5th toe', 'Left calcaneus', 'Right calcaneus',
    'Nose', 'Left eye', 'Right eye', 'Left ear', 'Right ear', 'Left shoulder', 'Right shoulder', 'Left elbow', 'Right elbow', 'Left wrist', 'Right wrist', 'Left 5th finger', 'Right 5th finger', 'Left 3rd finger', 'Right 3rd finger',
    'Left thumb', 'Right thumb', 'Left carpus', 'Right carpus'
    ]

#---Calculate angles and color data---
right_knee_angle = np.zeros(n_frames)
left_knee_angle = np.zeros(n_frames)
right_hip_angle = np.zeros(n_frames)
left_hip_angle = np.zeros(n_frames)

label_indices = [1,2,3,4,5,6,7,8,9,16,19,20,21,22,23,24,25,26]

joint_colors = ['k'] * n_joints

left_indices = [2, 5, 8, 19, 21, 23, 25]
right_indices = [3, 6, 9, 20, 22, 24, 26]

#Adjusting 0-based indices for Python
for idx in left_indices:
    joint_colors[idx - 1] = 'b'  # Left joints in blue
for idx in right_indices:
    joint_colors[idx - 1] = 'r'  # Right joints in red
    
#---Eksempel for frame---
frame_idx = 0 # Python uses 0-based indexing
x = joint_data[frame_idx, :, 0] #frontal
y = joint_data[frame_idx, :, 1] #Axial
z = joint_data[frame_idx, :, 2] #Sagittal

#---Static figure---
def expand_limits(data, margin=0.5):
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = max_val - min_val
    return (min_val - margin * range_val, max_val + margin * range_val)

#---Plot of figure---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')  # Proper 3D axis creation
           
ax.set_xlabel('X (frontal)')
ax.set_ylabel('Y (Axial)')
ax.set_zlabel('Z (Sagittal)')
ax.set_title('3D Joint Positions')
ax.view_init(elev=20.0, azim=30.0)
ax.grid(True)

#---Connect joints with lines---
def draw_line(indices, color='k', linewidth=2.0):
    ax.plot([x[i] for i in indices],
            [y[i] for i in indices],
            [z[i] for i in indices],
            color=color, linewidth=linewidth)

#Python uses 0-based indexing, Matlab uses 1-based indexing
#Basic skeleton connections
draw_line([8, 5, 2], 'r')  # Right leg 
draw_line([7, 4, 1], 'b')  # Left leg
draw_line([1, 0], 'b')  # Left hip
draw_line([2, 0], 'r')  # Right hip
draw_line([0, 3, 6], 'k')  # Spine
draw_line([21, 23, 25], 'r')  # Right arm
draw_line([20, 22, 24], 'b')  # Left arm
draw_line([6, 20], 'b')  # Left shoulder
draw_line([6, 21], 'r')  # Right shoulder

#Head connections
draw_line([16, 15], 'k', linewidth=1.5)  # Left eye -> Nose
draw_line([17, 15], 'k', linewidth=1.5)  # Right eye -> Nose
draw_line([16, 18], 'k', linewidth=1.5)  # Left eye -> Left ear
draw_line([17, 19], 'k', linewidth=1.5)  # Right eye -> Right ear
draw_line([15, 18], 'k', linewidth=1.5)  # Nose -> Left ear
draw_line([15, 19], 'k', linewidth=1.5)  # Nose -> Right ear
draw_line([18, 19], 'k', linewidth=1.5)  # Left ear -> Right ear

#Hand connections
draw_line([24, 26], 'b', linewidth=1.5)  # Left wrist -> Left 5th finger
draw_line([24, 28], 'b', linewidth=1.5)  # Left wrist -> Left 3rd finger
draw_line([24, 30], 'b', linewidth=1.5)  # Left wrist -> Left thumb
draw_line([24, 32], 'b', linewidth=1.5)  # Left wrist -> Left carpus

draw_line([25, 27], 'r', linewidth=1.5)  # Right wrist -> Right 5th finger
draw_line([25, 29], 'r', linewidth=1.5)  # Right wrist -> Right 3rd finger
draw_line([25, 31], 'r', linewidth=1.5)  # Right wrist -> Right thumb
draw_line([25, 33], 'r', linewidth=1.5)  # Right wrist -> Right carpus

#Feet connections
draw_line([7, 9], 'b', linewidth=1.5)  # Left ankle -> Left toe
draw_line([7, 11], 'b', linewidth=1.5)  # Left ankle -> Left 5th toe
draw_line([7, 13], 'b', linewidth=1.5)  # Left ankle -> Left calcaneus

draw_line([8, 10], 'r', linewidth=1.5)  # Right ankle -> Right toe
draw_line([8, 12], 'r', linewidth=1.5)  # Right ankle -> Right 5th toe
draw_line([8, 14], 'r', linewidth=1.5)  # Right ankle -> Right calcaneus

# Plot the joint points
ax.scatter(x, y, z, c=[joint_colors[i] for i in range(len(x))], s=50)

#---Labels---
for i in label_indices:
    idx = i - 1  # Adjust for 0-based indexing
    ax.text(x[idx], y[idx], z[idx], joint_names[idx],
            fontsize=8, color=joint_colors[idx], weight='bold')
    
#---Adjust axis limits---
ax.set_xlim(expand_limits(x))
ax.set_ylim(expand_limits(y))
ax.set_zlim(expand_limits(z))

# Simple test plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot all joints as points
scatter = ax.scatter(x, y, z, c=range(len(x)))  # Removed duplicate 's' parameter

ax.set_xlabel('X (frontal)')
ax.set_ylabel('Y (Axial)')
ax.set_zlabel('Z (Sagittal)')  # This works because ax is a 3D axis from projection='3d'
ax.set_title('3D Joint Positions - Test')

plt.show()

#---Hand selection---
def ask_hand_selection():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    choice = messagebox.askquestion(
        'Hand selection', 'Which hand i used for the movement? (Left or Right)'
    )
    if choice == 'yes':
        hand = 'Left'
    elif choice == 'no':
        hand = 'Right'
    else:
        raise ValueError("Invalid choice. Please select 'Left' or 'Right'.")
    return hand

#---Main code---
def main():
    n_frames = 1000
    frame_rate = 30
    time = np.linspace(0, n_frames / frame_rate, n_frames)
    np.random.seed(0)
    joint_data = np.random.rand(n_frames, 35, 3)
    
    choice = ask_hand_selection()
    
    if choice == 'Left':
        hand_label = 'Left hand'
        hand_joints = [24, 26, 28, 30, 32]
    elif choice == 'Right':
        hand_label = 'Right hand'
        hand_joints = [25, 27, 29, 31, 33]
    else:
        raise ValueError("Invalid choice. Please select 'Left' or 'Right'.")
    
    #Average vertical Y Position
    y_hand = np.mean(joint_data[:, hand_joints, 1], axis=1)

    #Ivert for valley detection
    inv_y = -y_hand

    #Find valleys (bottoms)
    bottoms, _ = find_peaks(inv_y, 
                            distance=round(0.05 * frame_rate),
                            prominence=0.5)
    #Find peaks (tops)
    tops, _ = find_peaks(y_hand, 
                          distance=round(0.05 * frame_rate),
                          prominence=0.5)

    #Ensure peak before first bottom
    if len(tops) == 0 or (len(bottoms) > 0 and bottoms[0] < tops[0]):
        tops = np.insert(tops, 0, 0)
    if len(bottoms) > 0 and bottoms[-1] < tops[-1]:
        tops = np.append(tops, n_frames - 1)
        
    #Match movement phases
    starts, ends = [], []
    for b in bottoms:
        prev_tops = tops[tops < b]
        next_tops = tops[tops > b]
        if len(prev_tops) > 0 and len(next_tops) > 0:
            starts.append(prev_tops[-1])
            ends.append(next_tops[0])

    starts = np.array(starts)
    ends = np.array(ends)

    print(f'Detected {len(bottoms)} full movement(s) based on {hand_label} Y position.')

    #---Plot---
    plt.figure(figsize=(15, 8))
    plt.plot(time, y_hand, 'b-', linewidth=1.5, label=hand_label)
    plt.plot(time[starts], y_hand[starts], 'go', markerfacecolor='k', markersize=8, label='Start (Top)')
    plt.plot(time[bottoms], y_hand[bottoms], 'rv', markerfacecolor='r', markersize=8, label='Bottom')
    plt.plot(time[ends], y_hand[ends], 'k^', markerfacecolor='k', markersize=8, label='End (Top)')

    for s, e in zip(starts, ends):
        plt.axvline(x=time[s], linestyle='--', color='k', linewidth=1.2)
        plt.axvline(x=time[e], linestyle='--', color='k', linewidth=1.2)
        
    plt.xlabel('Time (s)')
    plt.ylabel(f'{hand_label} Height (meters)')
    plt.title(f'Movement Detection Using {hand_label} trajectory')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    return starts, ends, time

if __name__ == '__main__':
    starts, ends, time = main()
    
#---Spine joints---
Lower_back = 0
Middle_back = 3
Upper_back = 6

#---Calculate angles---
spineAngle = np.zeros(n_frames)

print(f"Debug: n_frames = {n_frames}")
print(f"Debug: joint_data shape = {joint_data.shape}")

for f in range(n_frames):
    sacrum = joint_data[f, Lower_back, :]
    Thorax = joint_data[f, Middle_back, :]
    Neck = joint_data[f, Upper_back, :]
    
    vec1 = Thorax - sacrum
    vec2 = Neck - Thorax
    
    # Extract only Z (sagittal) and Y (axial) components for sagittal plane analysis
    vec1 = vec1[[2, 1]]  # [Z, Y]
    vec2 = vec2[[2, 1]]  # [Z, Y]
    
    # Calculate cross product in 2D (gives scalar)
    crossZ = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    dotProd = np.dot(vec1, vec2)
    
    # Calculate angle between vectors
    spineAngle[f] = -np.degrees(np.arctan2(crossZ, dotProd))

# Debug: Check array dimensions
print(f"Debug: spineAngle shape = {spineAngle.shape}")
print(f"Debug: time shape = {time.shape}")

# Fix: Ensure time array matches spineAngle array length
actual_frames = len(spineAngle)
time_spine = np.arange(actual_frames) / frame_rate

print(f"Debug: time_spine shape = {time_spine.shape}")

#---Define movement segments---
starts = [0]
ends = [actual_frames - 1]

#---Plot---
plt.figure(figsize=(15, 8))
plt.plot(time_spine, spineAngle, 'g-', linewidth=2, label='Spine Angle')

plt.xlabel('Time (s)')
plt.ylabel('Spine Angle (°)')
plt.title('Sagittal Spine Angle')
plt.grid(True) 

#---Y limits---
spMinAll = np.min(spineAngle)
spMaxAll = np.max(spineAngle)
margin = 0.1 * (spMaxAll - spMinAll)
plt.ylim([spMinAll - margin, spMaxAll + margin])

print('\nSagittal Spine ROM (deg):')
print('Movement\tStart\tEnd\tMin\tMax\tROM\tPosture')

for i in range(len(starts)):
    idxStart = starts[i]
    idxEnd = ends[i]
    
    if idxEnd > idxStart and idxEnd < len(spineAngle):
        sp = spineAngle[idxStart:idxEnd + 1]
        spMin = np.min(sp)
        spMax = np.max(sp)
        spROM = spMax - spMin
        
        # Determine posture based on angle values
        if spMin < -10:
            posture = 'Lordosis'
        elif spMax > 10:
            posture = 'Kyphosis'
        else:
            posture = 'Neutral'
        
        print(f'{i+1}\t\t{idxStart}\t{idxEnd}\t{spMin:.1f}\t{spMax:.1f}\t{spROM:.1f}\t{posture}')
        
        # Find indices of min and max values
        spMinIdx = idxStart + np.where(sp == spMin)[0][0]
        spMaxIdx = idxStart + np.where(sp == spMax)[0][0]
        
        # Plot min/max points
        plt.plot(time_spine[[spMinIdx, spMaxIdx]],
                 [spMin, spMax],
                 'ko', markerfacecolor='k', markersize=6)

        # Plot vertical lines for movement boundaries
        plt.axvline(time_spine[idxStart], linestyle='--', color='k', linewidth=1.2, alpha=0.7)
        plt.axvline(time_spine[idxEnd], linestyle='--', color='k', linewidth=1.2, alpha=0.7)

#---Manual labels---
xlims = plt.xlim()
xText = xlims[0] + 0.02 * (xlims[1] - xlims[0])

yFront = -30
yBack = 30

plt.text(xText, yFront, 'Lordosis (Extension)', 
         fontsize=11, fontweight='bold', color='b', 
         ha='left', va='top')

plt.text(xText, yBack, 'Kyphosis (Flexion)', 
         fontsize=11, fontweight='bold', color='r', 
         ha='left', va='bottom')

# Add reference line for neutral spine
plt.axhline(0, linestyle='--', color='k', linewidth=1.2, alpha=0.7)
plt.text(xlims[1] * 0.98, 0, 'Neutral Spine',
         va='bottom', ha='right', fontsize=10, fontweight='bold', color='k')

# Set consistent y-limits
plt.ylim([-45, 45])
plt.legend()
plt.tight_layout()
plt.show()

# Print summary statistics
print(f'\nOverall Statistics:')
print(f'Mean spine angle: {np.mean(spineAngle):.1f}°')
print(f'Standard deviation: {np.std(spineAngle):.1f}°')
print(f'Total range of motion: {np.max(spineAngle) - np.min(spineAngle):.1f}°')

#---Spine joint---
Lower_back = 0
Middle_back = 3
Upper_back = 6

#---Calculate angles---
spineAngle = np.zeros(n_frames)

for f in range(n_frames):
    sacrum = joint_data[f, Lower_back, :]
    Thorax = joint_data[f, Middle_back, :]
    Neck = joint_data[f, Upper_back, :]
    
    vec1 = Thorax - sacrum
    vec2 = Neck - Thorax
    
    #projection to frontal plane
    vec1 = vec1[[0, 1]]
    vec2 = vec2[[0, 1]]

    crossZ = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    dotProd = np.dot(vec1, vec2)
    
    spineAngle[f] = -np.degrees(np.arctan2(crossZ, dotProd))
    
#---Plot---
plt.figure(figsize=(15, 8))
plt.plot(time_spine, spineAngle, 'g-', linewidth=2, label='Spine Angle')

plt.xlabel('Time (s)')
plt.ylabel('Frontal Spine Angle (°)')
plt.title('Frontal Spine')
plt.grid(True)

#---Y limits---
spMinAll = np.min(spineAngle)
spMaxAll = np.max(spineAngle)
margin = 0.1 * (spMaxAll - spMinAll)
plt.ylim([spMinAll - margin, spMaxAll + margin])

print('\nSpine Lateral ROM (deg):')
print('Rep\tStart\tEnd\tMin\tMax\tROM')

for i in range(len(starts)):
    idxStart = starts[i]
    idxEnd = ends[i]
    
    if idxEnd > idxStart and idxEnd < len(spineAngle):
        sp = spineAngle[idxStart:idxEnd + 1]
        spMin = np.min(sp)
        spMax = np.max(sp)
        spROM = spMax - spMin
        
        if spMax > 5 and abs(spMax) > abs(spMin):
            posture = 'Right'
        elif spMin < -5 and abs(spMin) > abs(spMax):
            posture = 'Left'
        else:
            posture = 'Neutral'

        print(f'{i+1}\t{idxStart}\t{idxEnd}\t{spMin:.1f}\t{spMax:.1f}\t{spROM:.1f}')
        
        spMinIdx = idxStart + np.where(sp == spMin)[0][0]
        spMaxIdx = idxStart + np.where(sp == spMax)[0][0]
        
        plt.plot(time[[spMinIdx, spMaxIdx]],
                 [spMin, spMax],
                 'ko', markerfacecolor='k', markersize=6)
        
        plt.axvline(time[idxStart], linestyle='--', color='k')
        plt.axvline(time[idxEnd], linestyle='--', color='k')
        
#---Manual labels---
xlims = plt.xlim()
xText = xlims[0] + 0.02 * (xlims[1] - xlims[0])

yRight = -30
yLeft = 30

plt.text(xText, yFront, 'Right',
         color='k', fontsize=11, fontweight='bold',
         ha='left', va='top')

plt.text(xText, yBack, 'Left',
        color='k', fontsize=11, fontweight='bold',
        ha='left', va='bottom')

plt.axhline(0, linestyle='--', color='k', linewidth=1.2)
plt.text(xlims[1], 0, 'Midline spine',
         va='bottom', ha='left',
         fontsize=10, fontweight='bold', color='k')
         
plt.ylim([-45, 45])
plt.legend()
plt.show()

#Lateral spine flexion analysis (frontal plane)
#---Spine and hip joints---
LeftHip = 1
RightHip = 2
Lower_back = 0
Middle_back = 3

#---Calculate angles---
spineAngle = np.zeros(n_frames)

for f in range(n_frames):
    hipL = joint_data[f, LeftHip, :]
    hipR = joint_data[f, RightHip, :]
    hipVec = hipR - hipL
    
    lowback = joint_data[f, Lower_back, :]
    upperback = joint_data[f, Middle_back, :]
    spineVec = upperback - lowback
    
    hipXY = hipVec[[0, 1]]  # Frontal plane projection
    spineXY = spineVec[[0, 1]]  # Frontal plane projection
    
    dotProd = np.dot(hipXY, spineXY)
    normProd = np.linalg.norm(hipXY) * np.linalg.norm(spineXY)
    
    cosTheta = np.clip(dotProd / normProd, -1.0, 1.0)  # Ensure value is within valid range
    theta = np.degrees(np.arccos(cosTheta))
    spineAngle[f] = theta

#---90 degree rule---
spineLatAdjusted = spineAngle - 90

# Create time array with the same length as spineLatAdjusted
time = np.linspace(0, n_frames/frame_rate, n_frames)  # Adjust frame_rate as needed

plt.figure(figsize=(15, 8))
plt.plot(time, spineLatAdjusted, 'g-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Lateral Frontal Angle (°)')
plt.title('Spine Frontal Angle')
plt.grid(True)

plt.axhline(0, linestyle='--', color='k', linewidth=1.2)

xlims = plt.xlim()
plt.text(xlims[1] * 0.98, 0, 'Midline spine',
         va='bottom', ha='right',
            fontsize=10, fontweight='bold', color='k')

#---Y limits---
yMin = np.min(spineLatAdjusted)
yMax = np.max(spineLatAdjusted)
margin = 0.1 * (yMax - yMin)
plt.ylim([yMin - margin, yMax + margin])

print('\nSquat/Deadlift Frontal Spine ROM (deg from neutral):')
print('Rep\tStart\tEnd\tMin\tMax\tROM')

for i in range(len(starts)):
    s = starts[i]
    e = ends[i]
    
    if e > s and e < len(spineLatAdjusted):
        lat = spineLatAdjusted[s:e]
        latMin = np.min(lat)
        latMax = np.max(lat)
        latROM = latMax - latMin
        
        if latMax > 5 and abs(latMax) > abs(latMin):
            posture = 'Right'
        elif latMin < -5 and abs(latMin) > abs(latMax):
            posture = 'Left'
        else:
            posture = 'Neutral'
            
        print(f'{i+1}\t{s}\t{e}\t{latMin:.1f}\t{latMax:.1f}\t{latROM:.1f}')

        minIdx = s + np.where(lat == latMin)[0][0]
        maxIdx = s + np.where(lat == latMax)[0][0]
        
        plt.plot(time[[minIdx, maxIdx]], [latMin, latMax],
                 'ko', markerfacecolor='k')
        
        plt.axvline(time[s], linestyle='--', color='k', linewidth=1.2)
        plt.axvline(time[e], linestyle='--', color='k', linewidth=1.2)
        
#---Manual labels---
xlims = plt.xlim()
xText = xlims[0] + 0.02 * (xlims[1] - xlims[0])

yFront = -30
yBack = 30

plt.text(xText, yFront, 'Right',
            color='k', fontsize=11, fontweight='bold',
            ha='left', va='top')
plt.text(xText, yBack, 'Left',
            color='k', fontsize=11, fontweight='bold',
            ha='left', va='bottom')

plt.ylim([-60, 60])
plt.show()

#---Joint index---
LeftShoulder = 20
RightShoulder = 21
LeftHip = 1
RightHip = 2

shoulderRotationAngle = np.zeros(n_frames)

for f in range(n_frames):
    lShoulder = joint_data[f, LeftShoulder, :]
    rShoulder = joint_data[f, RightShoulder, :]
    lHip = joint_data[f, LeftHip, :]
    rHip = joint_data[f, RightHip, :]
    
    shoulderVec = rShoulder - lShoulder
    hipVec = rHip - lHip
    
    shoulderXZ = rShoulder - lShoulder
    hipXZ = rHip - lHip

    if np.linalg.norm(shoulderXZ) > 0 or np.linalg.norm(hipXZ) > 0:
        cosTheta = np.dot(shoulderXZ, hipXZ) / (np.linalg.norm(shoulderXZ) * np.linalg.norm(hipXZ))
        cosTheta = np.clip(cosTheta, -1.0, 1.0)
        theta = np.degrees(np.arccos(cosTheta))
    else:
        theta = np.nan
        
    #Correct angle direction
    crossVal = hipXZ[0] * shoulderXZ[1] - hipXZ[1] * shoulderXZ[0]
    if crossVal < 0:
        theta = -theta
        
    shoulderRotationAngle[f] = theta

# Create time array with the same length as shoulderRotationAngle
time = np.linspace(0, n_frames/frame_rate, n_frames)  # Adjust frame_rate as needed
    
#---Plot---
plt.figure(figsize=(15, 8))
plt.plot(time, shoulderRotationAngle, 'g-', linewidth=2, label='Shoulder Rotation Angle')
plt.xlabel('Time (s)')
plt.ylabel('Shoulder Rotation Angle (°)')
plt.title('Axial Trunk Rotation')
plt.grid(True)
plt.legend(loc='best')  # Fixed: was plt-legend

#---Horizontal line---
plt.axhline(0, linestyle='--', color='k', linewidth=1.2)

#---Y limits---
yMin = np.min(shoulderRotationAngle)
yMax = np.max(shoulderRotationAngle)
margin = 0.1 * (yMax - yMin)
plt.ylim([yMin - margin, yMax + margin])

print('\nShoulder Rotation ROM (deg):')
print('Rep\tStart\tEnd\tMin\tMax\tROM\tPosture')

for i in range(len(starts)):
    s = starts[i]
    e = ends[i]
    
    if e > s and e <= len(shoulderRotationAngle):
        rot = shoulderRotationAngle[s:e]
        rotMin = np.min(rot)
        rotMax = np.max(rot)
        rotROM = rotMax - rotMin
        
        if rotMax > 5 and abs(rotMax) > abs(rotMin):
            posture = 'Right Rotation'
        elif rotMin < -5 and abs(rotMin) > abs(rotMax):
            posture = 'Left Rotation'
        else:
            posture = 'Neutral'
        
        # Fixed variable names: was minRot, maxRot, romRot - now rotMin, rotMax, rotROM
        print(f'{i+1}\t{s}\t{e}\t{rotMin:.1f}\t{rotMax:.1f}\t{rotROM:.1f}\t{posture}')

        minIdx = s + np.where(rot == rotMin)[0][0]
        maxIdx = s + np.where(rot == rotMax)[0][0]
        
        plt.plot(time[[minIdx, maxIdx]], [rotMin, rotMax],
                 'ko', markerfacecolor='k', markersize=6)
        
        plt.axvline(time[s], linestyle='--', color='k', linewidth=1.2)
        plt.axvline(time[e], linestyle='--', color='k', linewidth=1.2)
        
#---Neutral line label---
xlims = plt.xlim()
plt.text(xlims[1], 0, 'Neutral Trunk',
         va='bottom', ha='right',
            fontsize=10, fontweight='bold', color='k')

#---Manual labels---
xText = xlims[0] + 0.02 * (xlims[1] - xlims[0])

yFront = -30
yBack = 30

plt.text(xText, yFront, 'Right Rotation',
         color='k', fontsize=11, fontweight='bold',
         ha='left', va='top')

plt.text(xText, yBack, 'Left Rotation',
         color='k', fontsize=11, fontweight='bold',
            ha='left', va='bottom')

plt.ylim([-45, 45])
plt.show()

#---Helper function---
def expand_limits(data, margin=0.1):
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = max_val - min_val
    return (min_val - margin * range_val, max_val + margin * range_val)

#---Animation function---
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib
import time

show_animation = input("Do you want to view the 3D joint animation? [y/n]: ").strip().lower()
if show_animation == 'y':

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3D Video')
    ax.set_xlabel('X (Frontal)')
    ax.set_ylabel('Y (Vertical)')
    ax.set_zlabel('Z (Sagittal)')
    ax.grid(True)

    scatter = ax.scatter([], [], [], s=60, c='k')

    # Labels
    texts = []
    for i in range(n_joints):
        if i in label_indices:
            t = ax.text(0, 0, 0, joint_names[i], fontsize=8,
                        color=joint_colors[i], weight='bold')
            texts.append(t)
        else:
            texts.append(None)

    # Lines
    def create_line(color='b', lw=2):
        return ax.plot([0, 0], [0, 0], [0, 0], color + '-', lw=lw)[0]

    hLeftLegLine = create_line('b')
    hRightLegLine = create_line('r')
    hLeftThighLine = create_line('b')
    hRightThighLine = create_line('r')
    hLeftPelvisLine = create_line('b')
    hRightPelvisLine = create_line('r')
    hSpineLine = ax.plot([0, 0, 0], [0, 0, 0], [0, 0, 0], 'k-', lw=2)[0]

    hLeftFoot1 = create_line('b', 1.5)
    hLeftFoot2 = create_line('b', 1.5)
    hLeftFoot3 = create_line('b', 1.2)
    hRightFoot1 = create_line('r', 1.5)
    hRightFoot2 = create_line('r', 1.5)
    hRightFoot3 = create_line('r', 1.2)

    hLeftArm = ax.plot([0, 0, 0], [0, 0, 0], [0, 0, 0], 'b-', lw=2)[0]
    hRightArm = ax.plot([0, 0, 0], [0, 0, 0], [0, 0, 0], 'r-', lw=2)[0]

    hSpineLeftShoulder = create_line('b')
    hSpineRightShoulder = create_line('r')

    hLeftHand = [create_line('b', 1.5) for _ in range(4)]
    hRightHand = [create_line('r', 1.5) for _ in range(4)]

    hHead = [create_line('k', 1.5) for _ in range(6)]
    hHead[5].set_linestyle('-')

    # Plot limits
    allX = joint_data[:, :, 0]
    allY = joint_data[:, :, 1]
    allZ = joint_data[:, :, 2]

    ax.set_xlim([np.min(allX), np.max(allX)])
    ax.set_ylim([np.min(allY), np.max(allY)])
    ax.set_zlim([np.min(allZ), np.max(allZ)])
    ax.view_init(elev=20., azim=-35)

    step = 2

    for frameIdx in range(0, n_frames, step):
        x = joint_data[frameIdx, :, 0]
        y = joint_data[frameIdx, :, 1]
        z = joint_data[frameIdx, :, 2]

        scatter._offsets3d = (x, y, z)

        for i in label_indices:
            if texts[i] is not None:
                texts[i].set_position((x[i], y[i]))
                texts[i].set_3d_properties(z[i])

        # Check if indices exist before accessing them
        def safe_line_update(line, indices, x, y, z):
            """Safely update line data if all indices are valid"""
            if all(idx < len(x) for idx in indices):
                line.set_data([x[idx] for idx in indices], [y[idx] for idx in indices])
                line.set_3d_properties([z[idx] for idx in indices])

        # Basic skeleton connections (only if indices exist)
        safe_line_update(hLeftLegLine, [7, 4], x, y, z)
        safe_line_update(hRightLegLine, [8, 5], x, y, z)
        safe_line_update(hLeftThighLine, [1, 4], x, y, z)
        safe_line_update(hRightThighLine, [2, 5], x, y, z)
        safe_line_update(hLeftPelvisLine, [1, 0], x, y, z)
        safe_line_update(hRightPelvisLine, [2, 0], x, y, z)

        # Spine (3 points)
        if all(idx < len(x) for idx in [0, 3, 6]):
            hSpineLine.set_data([x[0], x[3], x[6]], [y[0], y[3], y[6]])
            hSpineLine.set_3d_properties([z[0], z[3], z[6]])

        # Feet connections
        safe_line_update(hLeftFoot1, [7, 9], x, y, z)
        safe_line_update(hLeftFoot2, [7, 11], x, y, z)
        safe_line_update(hLeftFoot3, [7, 13], x, y, z)
        safe_line_update(hRightFoot1, [8, 10], x, y, z)
        safe_line_update(hRightFoot2, [8, 12], x, y, z)
        safe_line_update(hRightFoot3, [8, 14], x, y, z)

        # Arms (3 points each)
        if all(idx < len(x) for idx in [20, 22, 24]):
            hLeftArm.set_data([x[20], x[22], x[24]], [y[20], y[22], y[24]])
            hLeftArm.set_3d_properties([z[20], z[22], z[24]])

        if all(idx < len(x) for idx in [21, 23, 25]):
            hRightArm.set_data([x[21], x[23], x[25]], [y[21], y[23], y[25]])
            hRightArm.set_3d_properties([z[21], z[23], z[25]])

        # Hand connections - Fixed to avoid out of bounds access
        # Only connect if the target indices exist
        for j in range(4):
            left_hand_idx = 26 + j  # Changed from 24 + 2 * (j + 1) to avoid going beyond 29
            right_hand_idx = 26 + j  # Simplified hand joint mapping
            
            if 24 < len(x) and left_hand_idx < len(x):
                hLeftHand[j].set_data([x[24], x[left_hand_idx]], [y[24], y[left_hand_idx]])
                hLeftHand[j].set_3d_properties([z[24], z[left_hand_idx]])
            
            if 25 < len(x) and right_hand_idx < len(x):
                hRightHand[j].set_data([x[25], x[right_hand_idx]], [y[25], y[right_hand_idx]])
                hRightHand[j].set_3d_properties([z[25], z[right_hand_idx]])

        # Head connections
        safe_line_update(hHead[0], [16, 15], x, y, z)
        safe_line_update(hHead[1], [17, 15], x, y, z)
        safe_line_update(hHead[2], [16, 18], x, y, z)
        safe_line_update(hHead[3], [17, 19], x, y, z)
        safe_line_update(hHead[4], [15, 18], x, y, z)
        safe_line_update(hHead[5], [18, 19], x, y, z)

        # Shoulder connections
        safe_line_update(hSpineLeftShoulder, [6, 20], x, y, z)
        safe_line_update(hSpineRightShoulder, [6, 21], x, y, z)

        plt.pause(0.001)

    plt.show()
