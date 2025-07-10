import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Choose CSV-file ---
# Set default folder
default_folder = r'C:\Users\edm\OneDrive - Aalborg Universitet\ESA - E4D project\MoCap_Supporting_Files'

if not os.path.exists(default_folder):
    print('Warning: Default folder not found. Using current directory instead.')
    default_folder = os.getcwd()

# Use Tkinter to open file dialog
root = Tk()
root.withdraw()  # Hide main window
filename = filedialog.askopenfilename(
    initialdir=default_folder,
    title="Select CSV file with joint data",
    filetypes=[("CSV files", "*.csv")]
)

if not filename:
    print("User cancelled file selection.")
    exit()

print(f"Loading file: {filename}")
data = pd.read_csv(filename)

# --- Extract frame and joint data ---
n_joints = 34
n_frames = len(data)
joint_data = np.zeros((n_frames, n_joints, 3))

frame_rate = 30  # Hz
time_array = np.arange(n_frames) / frame_rate

for j in range(n_joints):
    joint_data[:, j, 0] = data[f'X{j}']  # Frontal
    joint_data[:, j, 1] = data[f'Y{j}']  # Axial
    joint_data[:, j, 2] = data[f'Z{j}']  # Sagittal

# --- Filter data ---
cutoff_freq = 1  # Hz
order = 4
nyquist_freq = frame_rate / 2
Wn = cutoff_freq / nyquist_freq
b, a = butter(order, Wn, btype='low')

joint_data_filtered = np.zeros_like(joint_data)
for j in range(n_joints):
    for dim in range(3):
        signal = joint_data[:, j, dim]
        joint_data_filtered[:, j, dim] = filtfilt(b, a, signal)

joint_data = joint_data_filtered

# --- Names of joints ---
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

# --- Prepare colors ---
label_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 19, 20, 21, 22, 23, 24, 25, 26]
joint_colors = ['k'] * n_joints  # set default color to black

left_indices = [2, 5, 8, 19, 21, 23, 25]   # 1-based from MATLAB
right_indices = [3, 6, 9, 20, 22, 24, 26]

# Adjust for 0-indexing in Python
for idx in left_indices:
    joint_colors[idx - 1] = 'b'
for idx in right_indices:
    joint_colors[idx - 1] = 'r'

def expand_limits(data, margin=0.05):
    """Expand axis limits with margin"""
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = max_val - min_val
    return (min_val - margin * range_val, max_val + margin * range_val)

def draw_skeleton_frame(frame_idx):
    """Draw skeleton for a single frame"""
    x = joint_data[frame_idx, :, 0]  # Frontal
    y = joint_data[frame_idx, :, 1]  # Axial
    z = joint_data[frame_idx, :, 2]  # Sagittal
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=60, c='k', marker='o')

    ax.set_xlabel('X (Frontal)')
    ax.set_ylabel('Y (Vertical)')
    ax.set_zlabel('Z (Sagittal)')
    ax.set_title(f'3D Joint Positions - Frame {frame_idx}')
    ax.view_init(elev=20, azim=30)
    ax.grid(True)

    def draw_line(indices, color='k', linewidth=2):
        """Draw line between joints"""
        if all(i < len(x) for i in indices):
            ax.plot([x[i] for i in indices],
                    [y[i] for i in indices],
                    [z[i] for i in indices],
                    color=color, linewidth=linewidth)

    # Draw skeleton connections
    draw_line([8, 5, 2], 'r')   # Right leg
    draw_line([7, 4, 1], 'b')   # Left leg
    draw_line([1, 0], 'b')      # Left hip
    draw_line([2, 0], 'r')      # Right hip
    draw_line([0, 3, 6], 'k')   # Spine
    draw_line([21, 23, 25], 'r')  # Right arm
    draw_line([20, 22, 24], 'b')  # Left arm
    draw_line([6, 20], 'b')     # Left shoulder
    draw_line([6, 21], 'r')     # Right shoulder

    # Head lines
    draw_line([16, 15], 'k', 1.5)  # Left Eye → Nose
    draw_line([17, 15], 'k', 1.5)  # Right Eye → Nose
    draw_line([16, 18], 'k', 1.5)  # Left Eye → Left Ear
    draw_line([17, 19], 'k', 1.5)  # Right Eye → Right Ear
    draw_line([15, 18], 'k', 1.5)  # Nose → Left Ear
    draw_line([15, 19], 'k', 1.5)  # Nose → Right Ear
    draw_line([18, 19], 'k', 1.2)  # Ear to Ear

    # Hands - Fixed indices
    if len(x) > 32:  # Check if we have hand data
        draw_line([24, 26], 'b', 1.5)  # Left wrist → 5th finger
        draw_line([24, 28], 'b', 1.5)  # Left wrist → 3rd finger
        draw_line([24, 30], 'b', 1.5)  # Left wrist → thumb
        draw_line([24, 32], 'b', 1.2)  # Left wrist → carpus

        draw_line([25, 27], 'r', 1.5)  # Right wrist → 5th finger
        draw_line([25, 29], 'r', 1.5)  # Right wrist → 3rd finger
        draw_line([25, 31], 'r', 1.5)  # Right wrist → thumb
        draw_line([25, 33], 'r', 1.2)  # Right wrist → carpus

    # Feet
    draw_line([7, 9], 'b', 1.5)    # Left ankle → toe
    draw_line([7, 11], 'b', 1.5)   # Left ankle → 5th toe
    draw_line([7, 13], 'b', 1.2)   # Left ankle → calcaneus

    draw_line([8, 10], 'r', 1.5)   # Right ankle → toe
    draw_line([8, 12], 'r', 1.5)   # Right ankle → 5th toe
    draw_line([8, 14], 'r', 1.2)   # Right ankle → calcaneus

    # Labels
    for i in label_indices:
        idx = i - 1  # 1-based to 0-based
        if idx < len(x):
            ax.text(x[idx], y[idx], z[idx], joint_names[idx],
                    fontsize=8, color=joint_colors[idx], weight='bold')

    # Expand axes for better visualization
    ax.set_xlim(expand_limits(x))
    ax.set_ylim(expand_limits(y))
    ax.set_zlim(expand_limits(z))

    plt.tight_layout()
    plt.show()

# Show first frame
draw_skeleton_frame(0)

# --- Animation section ---
def run_animation():
    """Run 3D animation of skeleton movement"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3D Animation')
    ax.set_xlabel('X (Frontal)')
    ax.set_ylabel('Y (Vertical)')
    ax.set_zlabel('Z (Sagittal)')
    ax.grid(True)

    # Set plot limits based on all data
    allX = joint_data[:, :, 0]
    allY = joint_data[:, :, 1]
    allZ = joint_data[:, :, 2]

    ax.set_xlim([np.min(allX), np.max(allX)])
    ax.set_ylim([np.min(allY), np.max(allY)])
    ax.set_zlim([np.min(allZ), np.max(allZ)])
    ax.view_init(elev=20., azim=-35)

    step = 2  # Skip frames for faster animation

    for frameIdx in range(0, n_frames, step):
        ax.clear()
        ax.set_xlabel('X (Frontal)')
        ax.set_ylabel('Y (Vertical)')
        ax.set_zlabel('Z (Sagittal)')
        ax.set_title(f'3D Animation - Frame {frameIdx}')
        ax.grid(True)
        ax.set_xlim([np.min(allX), np.max(allX)])
        ax.set_ylim([np.min(allY), np.max(allY)])
        ax.set_zlim([np.min(allZ), np.max(allZ)])
        ax.view_init(elev=20., azim=-35)
        
        x = joint_data[frameIdx, :, 0]
        y = joint_data[frameIdx, :, 1]
        z = joint_data[frameIdx, :, 2]

        # Plot joints
        ax.scatter(x, y, z, s=60, c='k', marker='o')

        # Draw skeleton connections (simplified for animation)
        def draw_line_anim(indices, color='k', linewidth=2):
            if all(i < len(x) for i in indices):
                ax.plot([x[i] for i in indices],
                        [y[i] for i in indices],
                        [z[i] for i in indices],
                        color=color, linewidth=linewidth)

        # Basic skeleton
        draw_line_anim([8, 5, 2], 'r')   # Right leg
        draw_line_anim([7, 4, 1], 'b')   # Left leg
        draw_line_anim([1, 0], 'b')      # Left hip
        draw_line_anim([2, 0], 'r')      # Right hip
        draw_line_anim([0, 3, 6], 'k')   # Spine
        draw_line_anim([21, 23, 25], 'r')  # Right arm
        draw_line_anim([20, 22, 24], 'b')  # Left arm
        draw_line_anim([6, 20], 'b')     # Left shoulder
        draw_line_anim([6, 21], 'r')     # Right shoulder

        plt.pause(0.01)  # Small pause for animation

    plt.show()

# Ask user for animation
show_animation = input("Do you want to view the 3D joint animation? [y/n]: ").strip().lower()
if show_animation == 'y':
    print("Animation starts...")
    run_animation()
else:
    print("Animation skipped.")

# === Squat/deadlift Detection Using Left Ear Vertical Movement ===
print("\n=== Starting Movement Detection ===")

leftEar = 18  # Left Ear index (0-based)
rightEar = 19  # Right Ear index (0-based)

# Check if ear indices are valid
if leftEar >= joint_data.shape[1] or rightEar >= joint_data.shape[1]:
    print(f"Error: Ear indices out of range. Max joint index: {joint_data.shape[1]-1}")
    leftEar = min(leftEar, joint_data.shape[1]-1)
    rightEar = min(rightEar, joint_data.shape[1]-1)
    print(f"Using adjusted indices: leftEar={leftEar}, rightEar={rightEar}")

yLeftEar = joint_data[:, leftEar, 1] 
yRightEar = joint_data[:, rightEar, 1]

# Initialize empty arrays for the case where no movements are detected
Starts = np.array([])
Ends = np.array([])
Bottoms = np.array([])
Tops = np.array([])

# Check for valid data
if np.all(np.isnan(yLeftEar)) or np.all(yLeftEar == 0):
    print("Warning: Left ear data appears to be invalid (all NaN or zeros)")
    print("Skipping movement detection...")
else:
    invY = -yLeftEar  

    # Find bottoms (squat/deadlift lowest points)
    try:
        _, Bottoms = find_peaks(invY,
                                distance=max(1, round(0.05 * frame_rate)),
                                prominence=0.05)
    except Exception as e:
        print(f"Error finding bottoms: {e}")
        Bottoms = np.array([])

# Find bottoms (squat/deadlift lowest points)
try:
    Bottoms, _ = find_peaks(invY,
                            distance=max(1, round(0.05 * frame_rate)),
                            prominence=0.05)
except Exception as e:
    print(f"Error finding bottoms: {e}")
    Bottoms = np.array([])

# Find tops
try:
    Tops, _ = find_peaks(yLeftEar,
                         distance=max(1, round(0.05 * frame_rate)),
                         prominence=0.05)
except Exception as e:
    print(f"Error finding tops: {e}")
    Tops = np.array([])

# Debug information
print(f"Number of peaks found: {len(Tops)}")
print(f"Number of bottoms found: {len(Bottoms)}")

# Only proceed if we have both tops and bottoms
if len(Tops) > 0 and len(Bottoms) > 0:
    # Top at start if necessary - FIXED: Now properly accessing array elements
    if Bottoms[0] < Tops[0]:
        Tops = np.insert(Tops, 0, 0)

    # Top at end if necessary
    if Bottoms[-1] > Tops[-1]:
        Tops = np.append(Tops, joint_data.shape[0] - 1)

    # Match squat/deadlift phases
    Starts = np.zeros_like(Bottoms, dtype=float)
    Ends = np.zeros_like(Bottoms, dtype=float)

    for i in range(len(Bottoms)):
        prev_top_indices = np.where(Tops < Bottoms[i])[0]
        next_top_indices = np.where(Tops > Bottoms[i])[0]

        if len(prev_top_indices) > 0 and len(next_top_indices) > 0:
            Starts[i] = Tops[prev_top_indices[-1]]
            Ends[i] = Tops[next_top_indices[0]]
        else:
            Starts[i] = np.nan
            Ends[i] = np.nan

    # Remove invalid entries
    valid = ~np.isnan(Starts) & ~np.isnan(Ends)
    Starts = Starts[valid].astype(int)
    Bottoms = Bottoms[valid].astype(int)
    Ends = Ends[valid].astype(int)
else:
    print("Warning: Insufficient peaks or bottoms detected for movement analysis")
    if len(Tops) == 0:
        print("No peaks (tops) found - try adjusting prominence or distance parameters")
    if len(Bottoms) == 0:
        print("No valleys (bottoms) found - try adjusting prominence or distance parameters")

    # Debug prints to check data
    print(f"Data shape: {joint_data.shape}")
    print(f"Time array length: {len(time_array)}")
    print(f"Left ear Y range: {yLeftEar.min():.3f} to {yLeftEar.max():.3f}")

    # === Plot left and right ear Y-trajectories with markers ===
    plt.figure(figsize=(12, 6))
    plt.plot(time_array, yLeftEar, 'b-', linewidth=1.5, label='Left Ear (Y)')
    plt.plot(time_array, yRightEar, 'r--', linewidth=1.5, label='Right Ear (Y)')

    # Only plot markers if we have valid detections
    if len(Starts) > 0:
        plt.plot(time_array[Starts], yLeftEar[Starts], 'go', markerfacecolor='g', markersize=8, label='Start (Top)')
        plt.plot(time_array[Bottoms], yLeftEar[Bottoms], 'rv', markerfacecolor='r', markersize=8, label='Bottom')
        plt.plot(time_array[Ends], yLeftEar[Ends], 'k^', markerfacecolor='k', markersize=8, label='End (Top)')
        
        # Add vertical lines for movement phases
        for i in range(len(Starts)):
            plt.axvline(x=time_array[Starts[i]], linestyle='--', color='g', alpha=0.7, linewidth=1)
            plt.axvline(x=time_array[Ends[i]], linestyle='--', color='k', alpha=0.7, linewidth=1)

    plt.title('Squat/Deadlift Detection using Left Ear Y-Trajectory')
    plt.xlabel('Time [s]')
    plt.ylabel('Vertical Position (Y)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Add text annotation if movements detected
    if len(Starts) > 0:
        ylims = plt.ylim()
        xlims = plt.xlim()
        plt.text(xlims[0] + 0.02 * (xlims[1] - xlims[0]),
                 ylims[0] + 0.05 * (ylims[1] - ylims[0]),
                 f'{len(Bottoms)} movements detected',
                 color='k',
                 fontweight='bold',
                 fontsize=11,
                 ha='left',
                 va='bottom')
    
    plt.show()

print("\nAnalysis complete!")

# === Sagittal Plane Knee Angles ===
rightHip, rightKnee, rightAnkle = 2, 5, 8  # Fixed: using 0-based indexing
leftHip, leftKnee, leftAnkle = 1, 4, 7     # Fixed: using 0-based indexing

nFrames = joint_data.shape[0]
rightKneeAngle = np.zeros(nFrames)
leftKneeAngle = np.zeros(nFrames)

for f in range(nFrames):
    # Extract sagittal plane (Z, Y)
    hipR = joint_data[f, rightHip, [2, 1]]
    kneeR = joint_data[f, rightKnee, [2, 1]]
    ankleR = joint_data[f, rightAnkle, [2, 1]]

    thighR = hipR - kneeR
    shankR = ankleR - kneeR
    
    # Handle zero vectors to avoid division by zero
    thigh_norm = np.linalg.norm(thighR)
    shank_norm = np.linalg.norm(shankR)
    
    if thigh_norm > 0 and shank_norm > 0:
        dot_product = np.dot(thighR, shankR) / (thigh_norm * shank_norm)
        # Clamp dot product to [-1, 1] to avoid numerical errors in arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        thetaR = np.degrees(np.arccos(dot_product))
        crossZ_R = thighR[0] * shankR[1] - thighR[1] * shankR[0]
        signR = np.sign(crossZ_R)
        rightKneeAngle[f] = signR * (180 - thetaR)
    else:
        rightKneeAngle[f] = 0  # Handle degenerate case
    
    # Left leg
    hipL = joint_data[f, leftHip, [2, 1]]
    kneeL = joint_data[f, leftKnee, [2, 1]]
    ankleL = joint_data[f, leftAnkle, [2, 1]]

    thighL = hipL - kneeL
    shankL = ankleL - kneeL
    
    # Handle zero vectors to avoid division by zero
    thigh_norm_L = np.linalg.norm(thighL)
    shank_norm_L = np.linalg.norm(shankL)
    
    if thigh_norm_L > 0 and shank_norm_L > 0:
        dot_product_L = np.dot(thighL, shankL) / (thigh_norm_L * shank_norm_L)
        # Clamp dot product to [-1, 1] to avoid numerical errors in arccos
        dot_product_L = np.clip(dot_product_L, -1.0, 1.0)
        
        thetaL = np.degrees(np.arccos(dot_product_L))
        crossZ_L = thighL[0] * shankL[1] - thighL[1] * shankL[0]
        signL = np.sign(crossZ_L)
        leftKneeAngle[f] = signL * (180 - thetaL)
    else:
        leftKneeAngle[f] = 0  # Handle degenerate case

# Print ROM per rep - only if we have detected movements
if len(Starts) > 0:
    print("\nSquat/deadlift Knee ROM (deg):")
    print("Rep\tStart\tEnd\tR_Min\tR_Max\tR_ROM\tL_Min\tL_Max\tL_ROM")

    for i in range(len(Starts)):
        s = Starts[i]
        e = Ends[i]
        r_min = np.min(rightKneeAngle[s:e+1])
        r_max = np.max(rightKneeAngle[s:e+1])
        r_rom = r_max - r_min
        l_min = np.min(leftKneeAngle[s:e+1])
        l_max = np.max(leftKneeAngle[s:e+1])
        l_rom = l_max - l_min
        print(f"{i+1}\t{s}\t{e}\t{r_min:.2f}\t{r_max:.2f}\t{r_rom:.2f}\t{l_min:.2f}\t{l_max:.2f}\t{l_rom:.2f}")
else:
    print("\nNo movements detected - cannot calculate ROM per repetition")

# === Plot the knee angles ===
plt.figure(figsize=(12, 6))
plt.plot(time_array, rightKneeAngle, 'r-', linewidth=1.5, label='Right Knee Angle')
plt.plot(time_array, leftKneeAngle, 'b-', linewidth=1.5, label='Left Knee Angle')

# Add horizontal line at 0 (neutral position)
plt.axhline(0, linestyle='--', color='k', linewidth=1.2)

# Add 'Knee extension' text on the right side
xlims = plt.xlim()
plt.text(xlims[1], 0, 'Knee extension',
         va='bottom', ha='left',
         fontsize=10, fontweight='bold', color='k')

# Mark repetitions if detected
if len(Starts) > 0:
    for i in range(len(Starts)):
        idxStart = Starts[i]
        idxEnd = Ends[i]
        
        if idxEnd > idxStart and idxEnd <= len(time_array):
            # Vertical lines for rep start/end
            plt.axvline(x=time_array[idxStart], linestyle='--', color='k', alpha=0.7)
            plt.axvline(x=time_array[idxEnd], linestyle='--', color='k', alpha=0.7)
            
            # Mark max/min points for each rep
            rk = rightKneeAngle[idxStart:idxEnd + 1]
            lk = leftKneeAngle[idxStart:idxEnd + 1]
            
            if len(rk) > 0 and len(lk) > 0:
                # Right knee max/min
                rkMax = np.max(rk)
                rkMin = np.min(rk)
                rkMaxIdx = idxStart + np.argmax(rk)
                rkMinIdx = idxStart + np.argmin(rk)
                
                # Left knee max/min
                lkMax = np.max(lk)
                lkMin = np.min(lk)
                lkMaxIdx = idxStart + np.argmax(lk)
                lkMinIdx = idxStart + np.argmin(lk)
                
                # Plot markers
                plt.plot(time_array[rkMaxIdx], rkMax, 'ro', markerfacecolor='r', markeredgecolor='k', markersize=6)
                plt.plot(time_array[rkMinIdx], rkMin, 'ro', markerfacecolor='r', markeredgecolor='k', markersize=6)
                plt.plot(time_array[lkMaxIdx], lkMax, 'bo', markerfacecolor='b', markeredgecolor='k', markersize=6)
                plt.plot(time_array[lkMinIdx], lkMin, 'bo', markerfacecolor='b', markeredgecolor='k', markersize=6)

# Final plot details
plt.xlabel('Time (s)')
plt.ylabel('Knee Angle (°)')
plt.title('Sagittal Plane Knee Angles')
plt.legend(['Right Knee', 'Left Knee'], loc='best')
plt.grid(True)

# Add text labels for flexion/extension
ylims = plt.ylim()
xlims = plt.xlim()

# Text positions
yFront = ylims[1] * 0.9  # Upper text
yBack = ylims[0] * 0.9   # Lower text
xText = xlims[0] + 0.02 * (xlims[1] - xlims[0])

# Text labels
plt.text(xText, yFront, 'Knee flexion',
         color='k', fontweight='bold', fontsize=11,
         ha='left', va='top')

plt.text(xText, yBack, 'Knee hyperextension',
         color='k', fontweight='bold', fontsize=11,
         ha='left', va='bottom')

plt.tight_layout()
plt.show()

# Plot knee angles
plt.figure(figsize=(12, 6))
plt.plot(time_array, rightKneeAngle, 'r-', linewidth=1.5, label='Right Knee Angle')
plt.plot(time_array, leftKneeAngle, 'b-', linewidth=1.5, label='Left Knee Angle')

# Add horizontal line at 0 (neutral position)
plt.axhline(0, linestyle='--', color='k', linewidth=1.2)

# Add 'Knee extension' text on the right side
xlims = plt.xlim()
plt.text(xlims[1], 0, 'Knee extension',
         va='bottom', ha='left',
         fontsize=10, fontweight='bold', color='k')

# Adjust y-limits based on both knee angles
yAll = np.concatenate([rightKneeAngle, leftKneeAngle])
yMin, yMax = np.min(yAll), np.max(yAll)
margin = 0.20 * (yMax - yMin)
plt.ylim([yMin - margin, yMax + margin])

# Loop over all detected reps
for i in range(len(Starts)):
    idxStart = Starts[i]
    idxEnd   = Ends[i]

    if idxEnd > idxStart and idxEnd <= len(time_array):
        rk = rightKneeAngle[idxStart:idxEnd + 1]
        lk = leftKneeAngle[idxStart:idxEnd + 1]

        # Right knee max/min
        rkMax = np.max(rk)
        rkMaxRelIdx = np.argmax(rk)
        rkMinRelIdx = rkMaxRelIdx + np.argmin(rk[rkMaxRelIdx + 1:]) + 1 if rkMaxRelIdx + 1 < len(rk) else rkMaxRelIdx
        rkMin = rk[rkMinRelIdx]
        rkROM = rkMax - rkMin

        # Left knee max/min
        lkMax = np.max(lk)
        lkMaxRelIdx = np.argmax(lk)
        lkMinRelIdx = lkMaxRelIdx + np.argmin(lk[lkMaxRelIdx + 1:]) + 1 if lkMaxRelIdx + 1 < len(lk) else lkMaxRelIdx
        lkMin = lk[lkMinRelIdx]
        lkROM = lkMax - lkMin

        print(f"{i+1}\t{idxStart}\t{idxEnd}\t{rkMin:.1f}\t{rkMax:.1f}\t{rkROM:.1f}\t{lkMin:.1f}\t{lkMax:.1f}\t{lkROM:.1f}")

        # Indices to original time series
        rkMaxIdx = idxStart + rkMaxRelIdx
        rkMinIdx = idxStart + rkMinRelIdx
        lkMaxIdx = idxStart + lkMaxRelIdx
        lkMinIdx = idxStart + lkMinRelIdx

        # Mark points without text
        plt.plot(time_array[[rkMaxIdx, rkMinIdx]], [rkMax, rkMin], 'ro', markerfacecolor='r', markeredgecolor='k')
        plt.plot(time_array[[lkMaxIdx, lkMinIdx]], [lkMax, lkMin], 'bo', markerfacecolor='b', markeredgecolor='k')

        # Vertical lines for rep start/end
        plt.axvline(x=time_array[idxStart], linestyle='--', color='k')
        plt.axvline(x=time_array[idxEnd], linestyle='--', color='k')

# Final plot details
plt.xlabel('Time (s)')
plt.ylabel('Knee Angle (°)')
plt.title('Knee flexion/extension')
plt.legend(['Right Knee', 'Left Knee'], loc='best')
plt.grid(True)

# Y and X limits for text placement
ylims = plt.ylim()
xlims = plt.xlim()

# Text positions
yFront = 130  # Top text
yBack = -25   # Bottom text
xText = xlims[0] + 0.02 * (xlims[1] - xlims[0])

# Text labels
plt.text(xText, yFront, 'Knee flexion',
         color='k', fontweight='bold', fontsize=11,
         ha='left', va='top')

plt.text(xText, yBack, 'Knee hyperextension',
         color='k', fontweight='bold', fontsize=11,
         ha='left', va='bottom')

# Fixed y-limits
plt.ylim([-30, 150])

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Antag nFrames, jointData og time er allerede defineret

# === Frontal Plane Knee Angles ===
rightHip, rightKnee, rightAnkle = 3, 6, 9
leftHip, leftKnee, leftAnkle = 2, 5, 8

rightKneeFrontalAngle = np.zeros(nFrames)
leftKneeFrontalAngle = np.zeros(nFrames)

for f in range(nFrames):
    # Right leg - frontal plane (X, Y)
    hipR = joint_data[f, rightHip, [0, 1]]  # Python 0-baseret indeks for X,Y er [0,1]
    kneeR = joint_data[f, rightKnee, [0, 1]]
    ankleR = joint_data[f, rightAnkle, [0, 1]]

    thighR = hipR - kneeR
    shankR = ankleR - kneeR

    # Vinklen i grader (acosd = arccos i grader)
    dot_prod_R = np.dot(thighR, shankR)
    norm_thighR = np.linalg.norm(thighR)
    norm_shankR = np.linalg.norm(shankR)
    cos_theta_R = dot_prod_R / (norm_thighR * norm_shankR)
    # Undgå små numeriske fejl som kan give værdier udenfor [-1,1]
    cos_theta_R = np.clip(cos_theta_R, -1.0, 1.0)
    thetaR = np.degrees(np.arccos(cos_theta_R))

    crossZ_R = thighR[0]*shankR[1] - thighR[1]*shankR[0]
    signAngleR = np.sign(crossZ_R)
    rightKneeFrontalAngle[f] = signAngleR * (180 - thetaR)

    # Left leg - frontal plane (X, Y)
    hipL = joint_data[f, leftHip, [0, 1]]
    kneeL = joint_data[f, leftKnee, [0, 1]]
    ankleL = joint_data[f, leftAnkle, [0, 1]]

    thighL = hipL - kneeL
    shankL = ankleL - kneeL

    dot_prod_L = np.dot(thighL, shankL)
    norm_thighL = np.linalg.norm(thighL)
    norm_shankL = np.linalg.norm(shankL)
    cos_theta_L = dot_prod_L / (norm_thighL * norm_shankL)
    cos_theta_L = np.clip(cos_theta_L, -1.0, 1.0)
    thetaL = np.degrees(np.arccos(cos_theta_L))

    crossZ_L = thighL[0]*shankL[1] - thighL[1]*shankL[0]
    signAngleL = np.sign(crossZ_L)
    leftKneeFrontalAngle[f] = signAngleL * (180 - thetaL)

# Vend fortegnet på højre knæ for at matche venstre ben
rightKneeFrontalAngle = -rightKneeFrontalAngle

# Print header
print('\nSquat/deadlift Knee ROM in Frontal Plane (deg):')
print('Rep\tStart\tEnd\tR_Min\tR_Max\tR_ROM\tL_Min\tL_Max\tL_ROM')

# Plot opsætning
plt.figure(figsize=(12, 6))
plt.plot(time_array, rightKneeFrontalAngle, 'r-', linewidth=1.5, label='Right Knee')
plt.plot(time_array, leftKneeFrontalAngle, 'b-', linewidth=1.5, label='Left Knee')

plt.xlabel('Time (s)')
plt.ylabel('Knee Angle (°)')
plt.title('Knee valgus/varus')
plt.legend(loc='best')
plt.grid(True)
plt.show()

ax = plt.gca()

# === Add horizontal line at 0 ===
ax.axhline(0, linestyle='--', color='k', linewidth=1.2, label='_nolegend_')  # '_nolegend_' skjuler linjen fra legend

# Get current axis limits
xlims = ax.get_xlim()
ylims = ax.get_ylim()

# Add 'knee midline' text label (bold)
ax.text(xlims[1], 0, 'knee midline',
        verticalalignment='bottom',
        horizontalalignment='left',
        fontsize=10,
        fontweight='bold',
        color='k')

# Manuelle y-positioner til labels
yFront = 70
yBack = -70

# X-position lidt ind fra venstre side
xText = xlims[0] + 0.02 * (xlims[1] - xlims[0])

# Tilføj labels ved faste Y positioner
ax.text(xText, yFront, 'Knee valgus',
        color='k',
        fontweight='bold',
        fontsize=11,
        horizontalalignment='left',
        verticalalignment='top')

ax.text(xText, yBack, 'Knee varus',
        color='k',
        fontweight='bold',
        fontsize=11,
        horizontalalignment='left',
        verticalalignment='bottom')

# Beregn y-limits med margin for bedre plot
yAll = np.concatenate((rightKneeFrontalAngle, leftKneeFrontalAngle))
yMin, yMax = yAll.min(), yAll.max()
margin = 0.20 * (yMax - yMin)
ax.set_ylim(yMin - margin, yMax + margin)

# Loop over reps og plot extrema + dividers + print ROM
for i in range(len(Starts)):
    idxStart = Starts[i]
    idxEnd = Ends[i]

    if idxEnd > idxStart and idxEnd <= len(time_array):
        rk = rightKneeFrontalAngle[idxStart:idxEnd+1]  # inklusiv idxEnd
        lk = leftKneeFrontalAngle[idxStart:idxEnd+1]

        # Right knee
        rkMax = np.max(rk)
        rkMin = np.min(rk)
        rkMaxRelIdx = np.where(rk == rkMax)[0][0]
        rkMinRelIdx = np.where(rk == rkMin)[0][0]
        rkROM = rkMax - rkMin

        # Left knee
        lkMax = np.max(lk)
        lkMin = np.min(lk)
        lkMaxRelIdx = np.where(lk == lkMax)[0][0]
        lkMinRelIdx = np.where(lk == lkMin)[0][0]
        lkROM = lkMax - lkMin

        # Print til console
        print(f"{i+1}\t{idxStart}\t{idxEnd}\t{rkMin:.1f}\t{rkMax:.1f}\t{rkROM:.1f}\t{lkMin:.1f}\t{lkMax:.1f}\t{lkROM:.1f}")

        # Find absolutte indeks i time-array
        rkMaxIdx = idxStart + rkMaxRelIdx
        rkMinIdx = idxStart + rkMinRelIdx
        lkMaxIdx = idxStart + lkMaxRelIdx
        lkMinIdx = idxStart + lkMinRelIdx

        # Markers (ingen tekst)
        ax.plot([time_array[rkMinIdx], time_array[rkMaxIdx]], [rkMin, rkMax], 'ko',
                markerfacecolor='r', markeredgecolor='k', label='_nolegend_')
        ax.plot([time_array[lkMinIdx], time_array[lkMaxIdx]], [lkMin, lkMax], 'ko',
                markerfacecolor='b', markeredgecolor='k', label='_nolegend_')

        # Dividers ved start og slut af squat/deadlift
        ax.axvline(time_array[idxStart], linestyle='--', color='k', label='_nolegend_')
        ax.axvline(time_array[idxEnd], linestyle='--', color='k', label='_nolegend_')

# adjust y-limits
ax.set_ylim(-90, 90)

plt.show()

# Definer viktige leddindekser (0-basert for Python)
Middle_back = 3
rightHip = 2
rightKnee = 5
leftHip = 1
leftKnee = 4

rightHipAngle = np.zeros(n_frames)
leftHipAngle = np.zeros(n_frames)

for f in range(n_frames):
    # Hent posisjoner (Z,Y)
    hipR = joint_data[f, rightHip, :]
    hipL = joint_data[f, leftHip, :]
    kneeR = joint_data[f, rightKnee, :]
    kneeL = joint_data[f, leftKnee, :]
    thorax = joint_data[f, Middle_back, :]

    trunkR = thorax[[2,1]] - hipR[[2,1]]  # (Z,Y)
    thighR = kneeR[[2,1]] - hipR[[2,1]]

    # Vinkel i grader med acosd
    cos_thetaR = np.dot(trunkR, thighR) / (np.linalg.norm(trunkR) * np.linalg.norm(thighR))
    cos_thetaR = np.clip(cos_thetaR, -1, 1)  # unngå numeriske problemer
    thetaR = np.degrees(np.arccos(cos_thetaR))

    crossZ_R = thighR[0]*trunkR[1] - thighR[1]*trunkR[0]
    signR = np.sign(crossZ_R)

    rightHipAngle[f] = signR * (180 - thetaR)

    trunkL = thorax[[2,1]] - hipL[[2,1]]
    thighL = kneeL[[2,1]] - hipL[[2,1]]

    cos_thetaL = np.dot(trunkL, thighL) / (np.linalg.norm(trunkL) * np.linalg.norm(thighL))
    cos_thetaL = np.clip(cos_thetaL, -1, 1)
    thetaL = np.degrees(np.arccos(cos_thetaL))

    crossZ_L = thighL[0]*trunkL[1] - thighL[1]*trunkL[0]
    signL = np.sign(crossZ_L)

    leftHipAngle[f] = signL * (180 - thetaL)

print("\nSquat/deadlift Hip ROM (deg):")
print("Rep\tStart\tEnd\tRightMin\tRightMax\tRightROM\tLeftMin\tLeftMax\tLeftROM")

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(time_array, rightHipAngle, 'r-', linewidth=2, label='Right Hip')
ax.plot(time_array, leftHipAngle, 'b-', linewidth=2, label='Left Hip')

# Add horizontal line at 0
ax.axhline(0, linestyle='--', color='k', linewidth=1.2, label='_nolegend_')

# Add 'Hip neutral' label
xlims = ax.get_xlim()
ax.text(xlims[1], 0, 'Hip neutral',
        verticalalignment='bottom',
        horizontalalignment='left',
        fontsize=10,
        fontweight='bold',
        color='k')

# Adjust y limits with margin
yAll = np.concatenate((rightHipAngle, leftHipAngle))
yMin, yMax = yAll.min(), yAll.max()
margin = 0.20 * (yMax - yMin)
ax.set_ylim(yMin - margin, yMax + margin)

for i in range(len(Starts)):
    idxStart = Starts[i]
    idxEnd = Ends[i]

    if idxEnd > idxStart and idxEnd < len(time_array):
        rh = rightHipAngle[idxStart:idxEnd+1]
        lh = leftHipAngle[idxStart:idxEnd+1]

        rhMax = np.max(rh)
        rhMaxRelIdx = np.argmax(rh)
        # Finn min etter maks
        if rhMaxRelIdx+1 < len(rh):
            rhMinRelIdx_offset = np.argmin(rh[rhMaxRelIdx+1:])
            rhMinRelIdx = rhMaxRelIdx + 1 + rhMinRelIdx_offset
        else:
            rhMinRelIdx = rhMaxRelIdx
        rhMin = rh[rhMinRelIdx]
        rhROM = rhMax - rhMin

        lhMax = np.max(lh)
        lhMaxRelIdx = np.argmax(lh)
        if lhMaxRelIdx+1 < len(lh):
            lhMinRelIdx_offset = np.argmin(lh[lhMaxRelIdx+1:])
            lhMinRelIdx = lhMaxRelIdx + 1 + lhMinRelIdx_offset
        else:
            lhMinRelIdx = lhMaxRelIdx
        lhMin = lh[lhMinRelIdx]
        lhROM = lhMax - lhMin

        print(f"{i+1}\t{idxStart}\t{idxEnd}\t{rhMin:.1f}\t{rhMax:.1f}\t{rhROM:.1f}\t{lhMin:.1f}\t{lhMax:.1f}\t{lhROM:.1f}")

        rhMaxIdx = idxStart + rhMaxRelIdx
        rhMinIdx = idxStart + rhMinRelIdx
        lhMaxIdx = idxStart + lhMaxRelIdx
        lhMinIdx = idxStart + lhMinRelIdx

        ax.plot([time_array[rhMaxIdx], time_array[rhMinIdx]], [rhMax, rhMin], 'ro', markerfacecolor='r', markeredgecolor='k', label='_nolegend_')
        ax.plot([time_array[lhMaxIdx], time_array[lhMinIdx]], [lhMax, lhMin], 'bo', markerfacecolor='b', markeredgecolor='k', label='_nolegend_')

        ax.axvline(time_array[idxStart], linestyle='--', color='k', label='_nolegend_')
        ax.axvline(time_array[idxEnd], linestyle='--', color='k', label='_nolegend_')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Hip Angle (°)')
ax.set_title('Hip flexion/extension')
ax.legend(loc='best')
ax.grid(True)

plt.show()# Definer viktige leddindekser (0-basert for Python)
Middle_back = 3
rightHip = 2
rightKnee = 5
leftHip = 1
leftKnee = 4

rightHipAngle = np.zeros(nFrames)
leftHipAngle = np.zeros(nFrames)

for f in range(nFrames):
    # Hent posisjoner (Z,Y)
    hipR = joint_data[f, rightHip, :]
    hipL = joint_data[f, leftHip, :]
    kneeR = joint_data[f, rightKnee, :]
    kneeL = joint_data[f, leftKnee, :]
    thorax = joint_data[f, Middle_back, :]

    trunkR = thorax[[2,1]] - hipR[[2,1]]  # (Z,Y)
    thighR = kneeR[[2,1]] - hipR[[2,1]]

    # Vinkel i grader med acosd
    cos_thetaR = np.dot(trunkR, thighR) / (np.linalg.norm(trunkR) * np.linalg.norm(thighR))
    cos_thetaR = np.clip(cos_thetaR, -1, 1)  # unngå numeriske problemer
    thetaR = np.degrees(np.arccos(cos_thetaR))

    crossZ_R = thighR[0]*trunkR[1] - thighR[1]*trunkR[0]
    signR = np.sign(crossZ_R)

    rightHipAngle[f] = signR * (180 - thetaR)

    trunkL = thorax[[2,1]] - hipL[[2,1]]
    thighL = kneeL[[2,1]] - hipL[[2,1]]

    cos_thetaL = np.dot(trunkL, thighL) / (np.linalg.norm(trunkL) * np.linalg.norm(thighL))
    cos_thetaL = np.clip(cos_thetaL, -1, 1)
    thetaL = np.degrees(np.arccos(cos_thetaL))

    crossZ_L = thighL[0]*trunkL[1] - thighL[1]*trunkL[0]
    signL = np.sign(crossZ_L)

    leftHipAngle[f] = signL * (180 - thetaL)

print("\nSquat/deadlift Hip ROM (deg):")
print("Rep\tStart\tEnd\tRightMin\tRightMax\tRightROM\tLeftMin\tLeftMax\tLeftROM")

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(time_array, rightHipAngle, 'r-', linewidth=2, label='Right Hip')
ax.plot(time_array, leftHipAngle, 'b-', linewidth=2, label='Left Hip')

# Add horizontal line at 0
ax.axhline(0, linestyle='--', color='k', linewidth=1.2, label='_nolegend_')

# Add 'Hip neutral' label
xlims = ax.get_xlim()
ax.text(xlims[1], 0, 'Hip neutral',
        verticalalignment='bottom',
        horizontalalignment='left',
        fontsize=10,
        fontweight='bold',
        color='k')

# Adjust y limits with margin
yAll = np.concatenate((rightHipAngle, leftHipAngle))
yMin, yMax = yAll.min(), yAll.max()
margin = 0.20 * (yMax - yMin)
ax.set_ylim(yMin - margin, yMax + margin)

for i in range(len(Starts)):
    idxStart = Starts[i]
    idxEnd = Ends[i]

    if idxEnd > idxStart and idxEnd < len(time_array):
        rh = rightHipAngle[idxStart:idxEnd+1]
        lh = leftHipAngle[idxStart:idxEnd+1]

        rhMax = np.max(rh)
        rhMaxRelIdx = np.argmax(rh)
        # Finn min etter maks
        if rhMaxRelIdx+1 < len(rh):
            rhMinRelIdx_offset = np.argmin(rh[rhMaxRelIdx+1:])
            rhMinRelIdx = rhMaxRelIdx + 1 + rhMinRelIdx_offset
        else:
            rhMinRelIdx = rhMaxRelIdx
        rhMin = rh[rhMinRelIdx]
        rhROM = rhMax - rhMin

        lhMax = np.max(lh)
        lhMaxRelIdx = np.argmax(lh)
        if lhMaxRelIdx+1 < len(lh):
            lhMinRelIdx_offset = np.argmin(lh[lhMaxRelIdx+1:])
            lhMinRelIdx = lhMaxRelIdx + 1 + lhMinRelIdx_offset
        else:
            lhMinRelIdx = lhMaxRelIdx
        lhMin = lh[lhMinRelIdx]
        lhROM = lhMax - lhMin

        print(f"{i+1}\t{idxStart}\t{idxEnd}\t{rhMin:.1f}\t{rhMax:.1f}\t{rhROM:.1f}\t{lhMin:.1f}\t{lhMax:.1f}\t{lhROM:.1f}")

        rhMaxIdx = idxStart + rhMaxRelIdx
        rhMinIdx = idxStart + rhMinRelIdx
        lhMaxIdx = idxStart + lhMaxRelIdx
        lhMinIdx = idxStart + lhMinRelIdx

        ax.plot([time_array[rhMaxIdx], time_array[rhMinIdx]], [rhMax, rhMin], 'ro', markerfacecolor='r', markeredgecolor='k', label='_nolegend_')
        ax.plot([time_array[lhMaxIdx], time_array[lhMinIdx]], [lhMax, lhMin], 'bo', markerfacecolor='b', markeredgecolor='k', label='_nolegend_')

        ax.axvline(time_array[idxStart], linestyle='--', color='k', label='_nolegend_')
        ax.axvline(time_array[idxEnd], linestyle='--', color='k', label='_nolegend_')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Hip Angle (°)')
ax.set_title('Hip flexion/extension')
ax.legend(loc='best')
ax.grid(True)

plt.show()

# Manuelle Y-posisjoner
yFront = 130   # Tekst øverst
yBack = -25    # Tekst nederst

# Finn x-akse grenser og regn ut litt innrykk
xlims = ax.get_xlim()
xText = xlims[0] + 0.02 * (xlims[1] - xlims[0])

# Legg til tekst på faste Y-posisjoner
ax.text(xText, yFront, 'Hip flexion',
        color='k',
        fontweight='bold',
        fontsize=11,
        horizontalalignment='left',
        verticalalignment='top')

ax.text(xText, yBack, 'Hip extension',
        color='k',
        fontweight='bold',
        fontsize=11,
        horizontalalignment='left',
        verticalalignment='bottom')

# Sett Y-akse begrensninger
ax.set_ylim([-30, 150])

plt.show()  # Vis figuren

Middle_back = 3     # MATLAB-indekser er 1-basert, Python 0-basert (så 4->3)
rightHip = 2
rightKnee = 5
leftHip = 1
leftKnee = 4

rightHipFrontalAngle = np.zeros(nFrames)
leftHipFrontalAngle = np.zeros(nFrames)

for f in range(nFrames):
    # Right hip - frontal plane (X-Y)
    thorax = joint_data[f, Middle_back, [0, 1]]  # [X, Y]
    hipR = joint_data[f, rightHip, [0, 1]]
    kneeR = joint_data[f, rightKnee, [0, 1]]

    trunkR = thorax - hipR  # Hip → thorax
    thighR = kneeR - hipR   # Hip → knee

    dotProdR = np.dot(trunkR, thighR)
    normProdR = np.linalg.norm(trunkR) * np.linalg.norm(thighR)
    thetaR = np.degrees(np.arccos(dotProdR / normProdR))

    crossZ_R = trunkR[0]*thighR[1] - trunkR[1]*thighR[0]
    signR = np.sign(crossZ_R)
    rightHipFrontalAngle[f] = signR * (180 - thetaR)

    # Left hip - frontal plane (X-Y)
    hipL = joint_data[f, leftHip, [0, 1]]
    kneeL = joint_data[f, leftKnee, [0, 1]]

    trunkL = thorax - hipL
    thighL = kneeL - hipL

    dotProdL = np.dot(trunkL, thighL)
    normProdL = np.linalg.norm(trunkL) * np.linalg.norm(thighL)
    thetaL = np.degrees(np.arccos(dotProdL / normProdL))

    crossZ_L = trunkL[0]*thighL[1] - trunkL[1]*thighL[0]
    signL = np.sign(crossZ_L)
    leftHipFrontalAngle[f] = signL * (180 - thetaL)

# Flip høyre side for anatomisk konvensjon om ønskelig
rightHipFrontalAngle = -rightHipFrontalAngle

# Plot
plt.figure(figsize=(12, 7))
plt.plot(time_array, rightHipFrontalAngle, 'r-', linewidth=2, label='Right Hip')
plt.plot(time_array, leftHipFrontalAngle, 'b-', linewidth=2, label='Left Hip')

plt.axhline(0, linestyle='--', color='k', linewidth=1.2)  # Nøytral linje

# Juster Y-akse margin
yAll = np.concatenate((rightHipFrontalAngle, leftHipFrontalAngle))
yMin, yMax = yAll.min(), yAll.max()
margin = 0.20 * (yMax - yMin)
plt.ylim(yMin - margin, yMax + margin)

print("\nSquat/deadlift Hip ROM (deg):")
print("Rep\tStart\tEnd\tRH_Min\tRH_Max\tRH_ROM\tLH_Min\tLH_Max\tLH_ROM")

for i in range(len(Starts)):
    idxStart = Starts[i]
    idxEnd = Ends[i]

    if idxEnd > idxStart and idxEnd <= len(time_array):
        rh = rightHipFrontalAngle[idxStart:idxEnd+1]  # +1 fordi python slicing ekskluderer siste
        lh = leftHipFrontalAngle[idxStart:idxEnd+1]

        # Høyre hofte
        rhMax = rh.max()
        rhMin = rh.min()
        rhMaxRelIdx = np.argmax(rh)
        rhMinRelIdx = np.argmin(rh)
        rhROM = rhMax - rhMin

        # Venstre hofte
        lhMax = lh.max()
        lhMin = lh.min()
        lhMaxRelIdx = np.argmax(lh)
        lhMinRelIdx = np.argmin(lh)
        lhROM = lhMax - lhMin

        print(f"{i+1}\t{idxStart}\t{idxEnd}\t{rhMin:.1f}\t{rhMax:.1f}\t{rhROM:.1f}\t{lhMin:.1f}\t{lhMax:.1f}\t{lhROM:.1f}")

        # Absolutte indekser
        rhMaxIdx = idxStart + rhMaxRelIdx
        rhMinIdx = idxStart + rhMinRelIdx
        lhMaxIdx = idxStart + lhMaxRelIdx
        lhMinIdx = idxStart + lhMinRelIdx

        # Markører (uten tekst)
        plt.plot(time_array[[rhMaxIdx, rhMinIdx]], [rhMax, rhMin], 'ro', markerfacecolor='r', markeredgecolor='k')
        plt.plot(time_array[[lhMaxIdx, lhMinIdx]], [lhMax, lhMin], 'bo', markerfacecolor='b', markeredgecolor='k')

        # Marker start og slutt på squat/deadlift
        plt.axvline(time_array[idxStart], linestyle='--', color='k')
        plt.axvline(time_array[idxEnd], linestyle='--', color='k')

plt.xlabel('Time (s)')
plt.ylabel('Hip Abduction/Adduction Angle (°)')
plt.title('Frontal Plane Hip Angles')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Antaget at en figur allerede er oprettet og plot er tegnet

plt.xlabel('Time (s)')
plt.ylabel('Hip Abduction/Adduction Angle (°)')
plt.title('Hip Abduction/Adduction (Frontal Plane)')
plt.legend(['Right Hip', 'Left Hip'], loc='best')
plt.grid(True)

# Hent nuværende x- og y-grænser
xlims = plt.xlim()
ylims = plt.ylim()

# Neutral label
plt.text(xlims[1], 0, 'Neutral hip',
         verticalalignment='bottom',
         horizontalalignment='left',
         fontsize=10,
         fontweight='bold',
         color='k')

# Manuel Y positioner
yFront = 80    # Tekst ved Y = 80 (top)
yBack = -85    # Tekst ved Y = -85 (bund)

# X position lidt inde fra venstre
xText = xlims[0] + 0.02 * (xlims[1] - xlims[0])

# Tilføj labels på faste Y positioner
plt.text(xText, yFront, 'Hip abduction',
         color='k',
         fontweight='bold',
         fontsize=11,
         horizontalalignment='left',
         verticalalignment='top')

plt.text(xText, yBack, 'Hip adduction',
         color='k',
         fontweight='bold',
         fontsize=11,
         horizontalalignment='left',
         verticalalignment='bottom')

# Juster y-aksegrænser som i MATLAB
plt.ylim([-90, 90])

plt.pause(0.001)  # pause for at opdatere plot - matplotlib har ikke direkte pause som MATLAB, 0.001 sek er nok
plt.close()       # lukker figuren

# Definer nøgleled
rightKnee, rightAnkle, rightToe = 6, 9, 11
leftKnee, leftAnkle, leftToe = 5, 8, 10

rightAnkleAngle = np.zeros(nFrames)
leftAnkleAngle = np.zeros(nFrames)

for f in range(nFrames):
    # RIGHT ANKLE (knee–ankle–toe)
    shankR = joint_data[f, rightAnkle, :] - joint_data[f, rightKnee, :]
    footR = joint_data[f, rightToe, :] - joint_data[f, rightAnkle, :]
    cos_thetaR = np.dot(shankR, footR) / (np.linalg.norm(shankR) * np.linalg.norm(footR))
    cos_thetaR = np.clip(cos_thetaR, -1, 1)  # for at undgå numeriske fejl udenfor domæne
    thetaR = np.degrees(np.arccos(cos_thetaR))
    rightAnkleAngle[f] = 180 - np.clip(thetaR, 0, 180)

    # LEFT ANKLE (knee–ankle–toe)
    shankL = joint_data[f, leftAnkle, :] - joint_data[f, leftKnee, :]
    footL = joint_data[f, leftToe, :] - joint_data[f, leftAnkle, :]
    cos_thetaL = np.dot(shankL, footL) / (np.linalg.norm(shankL) * np.linalg.norm(footL))
    cos_thetaL = np.clip(cos_thetaL, -1, 1)
    thetaL = np.degrees(np.arccos(cos_thetaL))
    leftAnkleAngle[f] = 180 - np.clip(thetaL, 0, 180)

print("\nSquat/deadlift Ankle ROM (deg):")
print("Rep\tStart\tEnd\tRightROM\tLeftROM")

plt.figure(figsize=(12, 7))
plt.plot(time_array, rightAnkleAngle, 'r-', linewidth=2, label='Right Ankle')
plt.plot(time_array, leftAnkleAngle, 'b-', linewidth=2, label='Left Ankle')

# Juster Y-akse med margin
yAll = np.concatenate([rightAnkleAngle, leftAnkleAngle])
yMin, yMax = np.min(yAll), np.max(yAll)
margin = 0.10 * (yMax - yMin)
plt.ylim(yMin - margin, yMax + margin)

for i in range(len(Starts)):
    idxStart = Starts[i]
    idxEnd = Ends[i]

    if idxEnd > idxStart and idxEnd <= len(time_array):
        ra = rightAnkleAngle[idxStart:idxEnd+1]
        la = leftAnkleAngle[idxStart:idxEnd+1]

        raMax = np.max(ra)
        raMin = np.min(ra)
        laMax = np.max(la)
        laMin = np.min(la)
        raROM = raMax - raMin
        laROM = laMax - laMin

        print(f"{i+1}\t{idxStart}\t{idxEnd}\t{raROM:.1f}\t\t{laROM:.1f}")

        # Find indekser for maks og min (brug np.argmax/argmin)
        raMaxIdx = idxStart + np.argmax(ra)
        raMinIdx = idxStart + np.argmin(ra)
        laMaxIdx = idxStart + np.argmax(la)
        laMinIdx = idxStart + np.argmin(la)

        # Marker toppunkter (uden legend)
        plt.plot(time_array[[raMaxIdx, raMinIdx]], [raMax, raMin], 'ro', markerfacecolor='r', markeredgecolor='k')
        plt.plot(time_array[[laMaxIdx, laMinIdx]], [laMax, laMin], 'bo', markerfacecolor='b', markeredgecolor='k')

        # Marker squat/deadlift start og slut
        plt.axvline(time_array[idxStart], linestyle='--', color='k')
        plt.axvline(time_array[idxEnd], linestyle='--', color='k')

plt.xlabel('Time (s)')
plt.ylabel('Ankle Angle (°)')
plt.title('Ankle (Dorsiflexion/Plantarflexion)')
plt.legend(loc='best')
plt.grid(True)

# Tilføj horisontal linje for neutral vinkel ved 90°
plt.axhline(90, linestyle='--', color='k', linewidth=1.2)
plt.text(plt.xlim()[0], 90, 'Neutral ankle alignment',
         verticalalignment='bottom',
         horizontalalignment='left')

# Manuel Y-positioner
yFront = 10
yBack = 170

# X-position lidt inde fra venstre
xlims = plt.xlim()
xText = xlims[0] + 0.02 * (xlims[1] - xlims[0])

plt.text(xText, yFront, 'Dorsiflexion',
         color='k', fontweight='bold', fontsize=11,
         horizontalalignment='left', verticalalignment='top')

plt.text(xText, yBack, 'Plantarflexion',
         color='k', fontweight='bold', fontsize=11,
         horizontalalignment='left', verticalalignment='bottom')

plt.ylim([0, 180])

plt.pause(0.001)
plt.close()

Lower_back, Middle_back, Upper_back = 1, 4, 7

spineAngle = np.zeros(nFrames)

for f in range(nFrames):
    sacrum = joint_data[f, Lower_back, :]    # base of spine
    thorax = joint_data[f, Middle_back, :]   # mid-spine
    neck = joint_data[f, Upper_back, :]      # upper spine

    vec1 = thorax - sacrum  # lower spine vector
    vec2 = neck - thorax    # upper spine vector

    # Projekt til sagittal plan (Z: fremad, Y: vertikal)
    vec1_2d = vec1[[2, 1]]  # numpy 0-baseret: 2=Z, 1=Y
    vec2_2d = vec2[[2, 1]]

    # Beregn signed vinkel med atan2 (radianer -> grader)
    crossZ = vec1_2d[0]*vec2_2d[1] - vec1_2d[1]*vec2_2d[0]
    dotProd = np.dot(vec1_2d, vec2_2d)
    angle_rad = -np.arctan2(crossZ, dotProd)  # NEGATIV for korrekt fortolkning
    spineAngle[f] = np.degrees(angle_rad)

plt.figure(figsize=(12,7))
plt.plot(time_array, spineAngle, 'g-', linewidth=2, label='Spine Angle')
plt.xlabel('Time (s)')
plt.ylabel('Spine Angle (°)')
plt.title('Sagittal Spine Angle')
plt.grid(True)

# Y-akse grænser med margin
spMinAll = np.min(spineAngle)
spMaxAll = np.max(spineAngle)
margin = 0.1 * (spMaxAll - spMinAll)
plt.ylim(spMinAll - margin, spMaxAll + margin)

print("\nSagittal Spine ROM (deg):")
print("Movement\tStart\tEnd\tMin\tMax\tROM")

for i in range(len(Starts)):
    idxStart = Starts[i]
    idxEnd = Ends[i]

    if idxEnd > idxStart and idxEnd <= len(spineAngle):
        sp = spineAngle[idxStart:idxEnd+1]
        spMin = np.min(sp)
        spMax = np.max(sp)
        spROM = spMax - spMin

        # Heuristik for klassificering
        if spMin < 10:
            posture = 'Lordosis'
        elif spMax > 40:
            posture = 'Kyphosis'
        else:
            posture = 'Normal'

        print(f"{i+1}\t{idxStart}\t{idxEnd}\t{spMin:.1f}\t{spMax:.1f}\t{spROM:.1f}")

        spMinIdx = idxStart + np.argmin(sp)
        spMaxIdx = idxStart + np.argmax(sp)

        # Marker minima og maxima
        plt.plot(time_array[[spMinIdx, spMaxIdx]], [spMin, spMax], 'ko',
                 markerfacecolor='g', markeredgecolor='k')

        # Lodrette linjer for start og slut
        plt.axvline(time_array[idxStart], linestyle='--', color='k')
        plt.axvline(time_array[idxEnd], linestyle='--', color='k')

# Tekstlabels og layout
xlims = plt.xlim()
xText = xlims[0] + 0.02 * (xlims[1] - xlims[0])

yFront, yBack = -30, 30

plt.text(xText, yFront, 'Lordosis',
         color='k', fontweight='bold', fontsize=11,
         horizontalalignment='left', verticalalignment='top')

plt.text(xText, yBack, 'Kyphosis',
         color='k', fontweight='bold', fontsize=11,
         horizontalalignment='left', verticalalignment='bottom')

# Neutral linje ved 0 grader
plt.axhline(0, linestyle='--', color='k', linewidth=1.2)

plt.text(xlims[1], 0, 'Straight spine',
         verticalalignment='bottom', horizontalalignment='left',
         fontsize=10, fontweight='bold', color='k')

plt.ylim([-45, 45])

plt.pause(0.001)
plt.close()

leftHip, rightHip = 2, 3
lowerBack, middleBack = 1, 4

spineLatAngle = np.zeros(nFrames)

for f in range(nFrames):
    hipL = joint_data[f, leftHip, :]
    hipR = joint_data[f, rightHip, :]
    hipVec = hipR - hipL

    lowBack = joint_data[f, lowerBack, :]
    upperB = joint_data[f, middleBack, :]
    spineVec = upperB - lowBack

    # Projektion på frontalplanet (X, Y)
    hipXY = hipVec[[0, 1]]     # indeks 0 = X, 1 = Y
    spineXY = spineVec[[0, 1]]

    # Vinkel mellem hip-linje og rygsøjlevektor
    cos_theta = np.dot(hipXY, spineXY) / (np.linalg.norm(hipXY) * np.linalg.norm(spineXY))
    # Sikkerhed mod numerisk fejl udenfor [-1,1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_theta))
    spineLatAngle[f] = theta

# Juster relativt til 90 grader (neutral)
spineLatAdjusted = spineLatAngle - 90

plt.figure(figsize=(12,7))
plt.plot(time_array, spineLatAdjusted, 'g-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Lateral Frontal angle (°)')
plt.title('Spine Frontal angle')
plt.grid(True)

# Neutral linje ved 0
plt.axhline(0, linestyle='--', color='k', linewidth=1.2)

plt.text(xlims[1], 0, 'Midline spine',
         verticalalignment='bottom',
         horizontalalignment='right',
         fontsize=10,
         fontweight='bold',
         color='k')

# Y-akse med margin
yMin = np.min(spineLatAdjusted)
yMax = np.max(spineLatAdjusted)
margin = 0.1 * (yMax - yMin)
plt.ylim(yMin - margin, yMax + margin)

print("\nSquat/deadlift Frontal Spine ROM (deg from neutral):")
print("Rep\tStart\tEnd\tMin\tMax\tROM")

for i in range(len(Starts)):
    s = Starts[i]
    e = Ends[i]

    if e > s and e <= len(spineLatAdjusted):
        lat = spineLatAdjusted[s:e+1]
        minLat = np.min(lat)
        maxLat = np.max(lat)
        romLat = maxLat - minLat

        # Dominant retning
        if maxLat > 5 and abs(maxLat) > abs(minLat):
            posture = 'Right'
        elif minLat < -5 and abs(minLat) > abs(maxLat):
            posture = 'Left'
        else:
            posture = 'Neutral'

        print(f"{i+1}\t{s}\t{e}\t{minLat:.1f}\t{maxLat:.1f}\t{romLat:.1f}")

        minIdx = s + np.argmin(lat)
        maxIdx = s + np.argmax(lat)

        plt.plot(time_array[[minIdx, maxIdx]], [minLat, maxLat], 'ko',
                 markerfacecolor='g', markeredgecolor='k')

        plt.axvline(time_array[s], linestyle='--', color='k')
        plt.axvline(time_array[e], linestyle='--', color='k')

# Tekstlabels venstre/højre
xText = xlims[0] + 0.02 * (xlims[1] - xlims[0])
yFront, yBack = -30, 30

plt.text(xText, yFront, 'Left',
         color='k',
         fontweight='bold',
         fontsize=11,
         horizontalalignment='left',
         verticalalignment='top')

plt.text(xText, yBack, 'Right',
         color='k',
         fontweight='bold',
         fontsize=11,
         horizontalalignment='left',
         verticalalignment='bottom')

plt.ylim([-45, 45])

plt.pause(0.001)
plt.close()

rightKnee, rightToe = 6, 11
leftKnee, leftToe = 5, 10

# Z-retning er indeks 2 i Python (0-baseret)
rightToeZ = joint_data[:, rightToe, 2] * 100
rightKneeZ = joint_data[:, rightKnee, 2] * 100
leftToeZ = joint_data[:, leftToe, 2] * 100
leftKneeZ = joint_data[:, leftKnee, 2] * 100

rightKneeOverToe = rightKneeZ - rightToeZ
leftKneeOverToe = leftKneeZ - leftToeZ

plt.figure(figsize=(12, 7))
plt.plot(time_array, rightKneeOverToe, 'r-', linewidth=1.5, label='Right Knee')
plt.plot(time_array, leftKneeOverToe, 'b-', linewidth=1.5, label='Left Knee')

plt.axhline(0, linestyle='--', color='k', linewidth=1.2)

plt.xlabel('Time (s)')
plt.ylabel('Knee - Toe Distance (cm)')
plt.title('Anterior Displacement of the knee')
plt.legend(loc='best')
plt.grid(True)

xlims = plt.xlim()
ylims = plt.ylim()

print("\nSquat/deadlift Knee Over Toe Distance:")
print("Rep\tR_Min\tR_Max\tR_AnyAhead\tL_Min\tL_Max\tL_AnyAhead")

for i in range(len(Starts)):
    idxStart = Starts[i]
    idxEnd = Ends[i]

    rkDist = rightKneeOverToe[idxStart:idxEnd+1]  # +1 for inclusive indexing
    lkDist = leftKneeOverToe[idxStart:idxEnd+1]

    rkMin = np.min(rkDist)
    rkMax = np.max(rkDist)
    rkForward = np.any(rkDist > 0)

    lkMin = np.min(lkDist)
    lkMax = np.max(lkDist)
    lkForward = np.any(lkDist > 0)

    print(f"{i+1}\t{rkMin:.2f}\t{rkMax:.2f}\t{rkForward}\t{lkMin:.2f}\t{lkMax:.2f}\t{lkForward}")

    plt.axvline(time_array[idxStart], linestyle='--', color='k')
    plt.axvline(time_array[idxEnd], linestyle='--', color='k')

    rkMaxIdx = idxStart + np.argmax(rkDist)
    rkMinIdx = idxStart + np.argmin(rkDist)
    lkMaxIdx = idxStart + np.argmax(lkDist)
    lkMinIdx = idxStart + np.argmin(lkDist)

    plt.plot(time_array[[rkMaxIdx, rkMinIdx]], [rkMax, rkMin], 'ro', markerfacecolor='k')
    plt.plot(time_array[[lkMaxIdx, lkMinIdx]], [lkMax, lkMin], 'bo', markerfacecolor='k')

plt.text(xlims[1], 0, 'Knee on top of the toe',
         verticalalignment='bottom',
         horizontalalignment='left',
         fontsize=10,
         fontweight='bold',
         color='k')

yFront, yBack = 25, -25
xText = xlims[0] + 0.02 * (xlims[1] - xlims[0])

plt.text(xText, yFront, 'Knee in front of the toe',
         color='k',
         fontweight='bold',
         fontsize=11,
         horizontalalignment='left',
         verticalalignment='top')

plt.text(xText, yBack, 'Knee behind the toe',
         color='k',
         fontweight='bold',
         fontsize=11,
         horizontalalignment='left',
         verticalalignment='bottom')

plt.ylim([-30, 30])

plt.pause(0.001)
plt.close()

Lower_back = 1
headCluster = [16, 17, 18, 19, 20]  # Nose, Eyes, Ears
leftShoulder = 21
rightShoulder = 22
leftHip = 2
rightHip = 3

sacrumHeadDist = np.zeros(nFrames)
leftShoulderHipDist = np.zeros(nFrames)
rightShoulderHipDist = np.zeros(nFrames)

for f in range(nFrames):
    sacrumPos = joint_data[f, Lower_back, :]
    headPos = np.mean(joint_data[f, headCluster, :], axis=0)
    sacrumHeadDist[f] = np.linalg.norm(headPos - sacrumPos) * 100

    lShoulder = joint_data[f, leftShoulder, :]
    lHip = joint_data[f, leftHip, :]
    rShoulder = joint_data[f, rightShoulder, :]
    rHip = joint_data[f, rightHip, :]

    leftShoulderHipDist[f] = np.linalg.norm(lShoulder - lHip) * 100
    rightShoulderHipDist[f] = np.linalg.norm(rShoulder - rHip) * 100

plt.figure(figsize=(12, 7))
plt.plot(time_array, sacrumHeadDist, 'm-', linewidth=1.5, label='Head–Low back')
plt.plot(time_array, leftShoulderHipDist, 'b-', linewidth=1.5, label='Left Shoulder–Hip')
plt.plot(time_array, rightShoulderHipDist, 'r-', linewidth=1.5, label='Right Shoulder–Hip')

plt.xlabel('Time (s)')
plt.ylabel('Distance (cm)')
plt.title('Segment Distances: Head–Pelvis and Shoulder–Hip')
plt.legend(loc='best')
plt.grid(True)

allY = np.concatenate([sacrumHeadDist, leftShoulderHipDist, rightShoulderHipDist])
ymin = allY.min()
ymax = allY.max()
margin = 0.1 * (ymax - ymin)
plt.ylim(ymin - margin, ymax + margin)

for i in range(len(Starts)):
    plt.axvline(time_array[Starts[i]], linestyle='--', color='k')
    plt.axvline(time_array[Ends[i]], linestyle='--', color='k')

print("\nPer-movement Segment Distance ROM (cm):")
print("Movement\tStart\tEnd\tHeadMin\tHeadMax\tHeadROM\tLHipMin\tLHipMax\tLHipROM\tRHipMin\tRHipMax\tRHipROM")

for i in range(len(Starts)):
    idxStart = Starts[i]
    idxEnd = Ends[i]

    h = sacrumHeadDist[idxStart:idxEnd+1]
    l = leftShoulderHipDist[idxStart:idxEnd+1]
    r = rightShoulderHipDist[idxStart:idxEnd+1]

    hMin, hMax = h.min(), h.max()
    hROM = hMax - hMin
    lMin, lMax = l.min(), l.max()
    lROM = lMax - lMin
    rMin, rMax = r.min(), r.max()
    rROM = rMax - rMin

    print(f"{i+1}\t{idxStart}\t{idxEnd}\t{hMin:.2f}\t{hMax:.2f}\t{hROM:.2f}\t"
          f"{lMin:.2f}\t{lMax:.2f}\t{lROM:.2f}\t{rMin:.2f}\t{rMax:.2f}\t{rROM:.2f}")

plt.ylim(30, 70)

plt.pause(0.001)
plt.close()
# %% Save output
n = len(Starts)

# Opret DataFrame svarende til MATLAB table
results = pd.DataFrame({
    'squat/deadlift': np.arange(1, n+1),
    'StartFrame': Starts,
    'EndFrame': Ends
})

# Initialiser arrays til minimum, maksimum og ROM (range of motion)
raMin = np.zeros(n)
raMax = np.zeros(n)
raROM = np.zeros(n)

laMin = np.zeros(n)
laMax = np.zeros(n)
laROM = np.zeros(n)

rkMin = np.zeros(n)
rkMax = np.zeros(n)
rkROM = np.zeros(n)

lkMin = np.zeros(n)
lkMax = np.zeros(n)
lkROM = np.zeros(n)

rhMin = np.zeros(n)
rhMax = np.zeros(n)
rhROM = np.zeros(n)

lhMin = np.zeros(n)
lhMax = np.zeros(n)
lhROM = np.zeros(n)

spineMin = np.zeros(n)
spineMax = np.zeros(n)
spineROM = np.zeros(n)

snMin = np.zeros(n)
snMax = np.zeros(n)
snROM = np.zeros(n)

for i in range(n):
    s = Starts[i]
    e = Ends[i]

    # Spine
    sp = spineAngle[s:e+1]
    spineMin[i] = sp.min()
    spineMax[i] = sp.max()
    spineROM[i] = spineMax[i] - spineMin[i]

    # Knees
    rk = rightKneeAngle[s:e+1]
    rkMin[i] = rk.min()
    rkMax[i] = rk.max()
    rkROM[i] = rkMax[i] - rkMin[i]

    lk = leftKneeAngle[s:e+1]
    lkMin[i] = lk.min()
    lkMax[i] = lk.max()
    lkROM[i] = lkMax[i] - lkMin[i]

    # Hips
    rh = rightHipAngle[s:e+1]
    rhMin[i] = rh.min()
    rhMax[i] = rh.max()
    rhROM[i] = rhMax[i] - rhMin[i]

    lh = leftHipAngle[s:e+1]
    lhMin[i] = lh.min()
    lhMax[i] = lh.max()
    lhROM[i] = lhMax[i] - lhMin[i]

    # Ankles
    ra = rightAnkleAngle[s:e+1]
    raMin[i] = ra.min()
    raMax[i] = ra.max()
    raROM[i] = raMax[i] - raMin[i]

    la = leftAnkleAngle[s:e+1]
    laMin[i] = la.min()
    laMax[i] = la.max()
    laROM[i] = laMax[i] - laMin[i]

    # Sacrum–Head Distance
    d = sacrumHeadDist[s:e+1]
    snMin[i] = d.min()
    snMax[i] = d.max()
    snROM[i] = snMax[i] - snMin[i]

# Tilføj kolonner til DataFrame
results['Spine_Min'] = spineMin
results['Spine_Max'] = spineMax
results['Spine_ROM'] = spineROM

results['RKnee_Min'] = rkMin
results['RKnee_Max'] = rkMax
results['RKnee_ROM'] = rkROM

results['LKnee_Min'] = lkMin
results['LKnee_Max'] = lkMax
results['LKnee_ROM'] = lkROM

results['RHip_Min'] = rhMin
results['RHip_Max'] = rhMax
results['RHip_ROM'] = rhROM

results['LHip_Min'] = lhMin
results['LHip_Max'] = lhMax
results['LHip_ROM'] = lhROM

results['RAnkle_Min'] = raMin
results['RAnkle_Max'] = raMax
results['RAnkle_ROM'] = raROM

results['LAnkle_Min'] = laMin
results['LAnkle_Max'] = laMax
results['LAnkle_ROM'] = laROM

results['SacrumNose_Min'] = snMin
results['SacrumNose_Max'] = snMax
results['SacrumNose_ROM'] = snROM

# === Save to Excel in same folder as input CSV ===
filename_without_ext = os.path.splitext(filename)[0]
output_dir = os.path.dirname(filename)
outputFile = os.path.join(output_dir, f'analysis_output_{os.path.basename(filename_without_ext)}.xlsx')

# Gem DataFrame til Excel
results.to_excel(outputFile, index=False)

print(f'✅ Excel file saved: {outputFile}')
