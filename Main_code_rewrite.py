import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from scipy.signal import find_peaks

# Load CSV files 
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

if not file_path:
    print("User canceled file selection.")
    exit()

print(f"Loading data from: {file_path}")
data = pd.read_csv(file_path)

# Setup
n_joints = 34
n_frames = len(data)
frame_rate = 100 # Hz
time = np.arange(n_frames) / frame_rate

# Extract joint positions 3D
joint_data = np.zeros((n_frames, n_joints, 3))
for j in range(n_joints):
    joint_data[:, j, 0] = data[f"X{j}"]
    joint_data[:, j, 1] = data[f"Y{j}"]
    joint_data[:, j, 2] = data[f"Z{j}"]

required_columns = [f"{axis}{j}" for j in range(n_joints) for axis in ['X', 'Y', 'Z']]
if not all(col in data.columns for col in required_columns):
    print("Error: Missing required columns in the CSV file.")
    exit() 

# Squat detection using left ear (Y-coordinate)
left_ear_index = 18 # zero-indexed
y_ear = joint_data[:, left_ear_index, 1] # Y = vertical

# Find squat bottoms (valleys)
prominence_value = 0.02
inv_y = -y_ear
squat_bottoms, _ = find_peaks(inv_y, distance=int(0.05 * frame_rate), prominence=prominence_value)
print("Squat bottoms (frames):", squat_bottoms)

# Find squat tops (peaks)
squat_tops, _ = find_peaks(y_ear, distance=int(0.05 * frame_rate), prominence=prominence_value)
print("Squat tops (frames):", squat_tops)

# If no tops or bottoms are detected, exit with a message
if len(squat_bottoms) == 0 or len(squat_tops) == 0:
    print("No squats detected. Please check the input data or adjust detection parameters.")
    print(f"y_ear data: {y_ear}")
    exit()

# Ensure squat_tops and squat_bottoms are properly aligned
if squat_bottoms[0] < squat_tops[0]:
    squat_tops = np.insert(squat_tops, 0, 0)
if squat_bottoms[-1] > squat_tops[-1]:
    squat_tops = np.append(squat_tops, n_frames - 1)

# Pair squat bottoms with nearest tops
squat_starts = []
squat_ends = []
for bottom in squat_bottoms:
    prev_tops = squat_tops[squat_tops < bottom]
    next_tops = squat_tops[squat_tops > bottom]
    if len(prev_tops) > 0 and len(next_tops) > 0:
        squat_starts.append(prev_tops[-1])
        squat_ends.append(next_tops[0])

# Add a check to ensure squat_bottoms and squat_tops are not empty
if len(squat_bottoms) == 0 or len(squat_tops) == 0:
    print("No squats detected. Please check the input data.")
    exit()

# Ensure squat_tops and squat_bottoms are properly aligned
if len(squat_tops) == 0 or squat_bottoms[0] < squat_tops[0]:
    squat_tops = np.insert(squat_tops, 0, 0)
if squat_bottoms[-1] > squat_tops[-1]:
    squat_tops = np.append(squat_tops, n_frames - 1)

# Arrays
squat_starts = np.array(squat_starts)
squat_bottoms = np.array(squat_bottoms[:len(squat_starts)])
squat_ends = np.array(squat_ends)

print(f"Detected {len(squat_bottoms)} full squats based on left ear Y-coordinate.")

# Plot for visual check
plt.figure()
plt.plot(time, y_ear, 'b-', label='Left Ear Y')
plt.plot(time[squat_starts], y_ear[squat_starts], 'go', label='Start (Top)')
plt.plot(time[squat_bottoms], y_ear[squat_bottoms], 'rv', label='Bottom')
plt.plot(time[squat_ends], y_ear[squat_ends], 'k^', label='End (Top)')

for start, bottom, end in zip(squat_starts, squat_bottoms, squat_ends):
    plt.annotate(f"{y_ear[start]:.2f}", xy=(time[start], y_ear[start]), xytext=(time[start], y_ear[start] + 0.05),
                 textcoords='offset points', fontsize=8, color='green', ha='center')
    plt.annotate(f"{y_ear[bottom]:.2f}", xy=(time[bottom], y_ear[bottom]), xytext=(time[bottom], y_ear[bottom] - 0.05),
                 textcoords='offset points', fontsize=8, color='red', ha='center')
    plt.annotate(f"{y_ear[end]:.2f}", xy=(time[end], y_ear[end]), xytext=(time[end], y_ear[end] + 0.05),
                 textcoords='offset points', fontsize=8, color='black', ha='center')

    plt.axvline(time[start], color='g', linestyle='--', linewidth=1)
    plt.axvline(time[end], color='k', linestyle='--', linewidth=1)
# Annotate top and bottom points
for start, bottom, end in zip(squat_starts, squat_bottoms, squat_ends):
    plt.text(time[start], y_ear[start], f"{y_ear[start]:.2f}", color='green', fontsize=8, ha='center')
    plt.text(time[bottom], y_ear[bottom], f"{y_ear[bottom]:.2f}", color='red', fontsize=8, ha='center')
    plt.text(time[end], y_ear[end], f"{y_ear[end]:.2f}", color='black', fontsize=8, ha='center')

    plt.axvline(time[start], color='g', linestyle='--', linewidth=1)
    plt.axvline(time[end], color='k', linestyle='--', linewidth=1)

plt.xlabel("Time (seconds)")
plt.ylabel("Left Ear Height (Y)")
plt.title("Squat Detection")
plt.legend()
plt.grid(True)
plt.show()

# Body angles calculation
def calculate_angle(a, b, c):
    """Calculate the angle between point b and c in 3D space."""
    ab = a - b
    cb = c - b
    cos_theta = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure value is in valid range¨
    return np.degrees(np.arccos(cos_theta))

# Joint indices for angles
idx = {
    'Lower_back': 0,
    'Middle_back': 3,
    'Upper_back': 6,
    'rightHip': 2,
    'rightKnee': 5,
    'rightAnkle': 8,
    'rightToe': 10,
    'rightCalc': 14,
    'leftHip': 1,
    'leftKnee': 4,
    'leftAnkle': 7,
    'leftToe': 9,
    'leftCalc': 13,
    }

# Angle arrays
right_knee_angle = np.zeros(n_frames)
left_knee_angle = np.zeros(n_frames)
right_hip_angle = np.zeros(n_frames)
left_hip_angle = np.zeros(n_frames)
right_ankle_angle = np.zeros(n_frames)
left_ankle_angle = np.zeros(n_frames)

# Compute angles frame by frame
for f in range(n_frames):
    # Right leg
    hip = joint_data[f, idx['rightHip']]
    knee = joint_data[f, idx['rightKnee']]
    ankle = joint_data[f, idx['rightAnkle']]
    toe = joint_data[f, idx['rightToe']]
    calc = joint_data[f, idx['rightCalc']]
    thorax = joint_data[f, idx['Middle_back']]
    midfoot = (toe + calc) / 2

    right_knee_angle[f] = 180 - calculate_angle(hip, knee, ankle)
    right_hip_angle[f] = 180 - calculate_angle(thorax, hip, knee)
    right_ankle_angle[f] = 180 - calculate_angle(knee, ankle, midfoot)
   
    # Left leg
    hip = joint_data[f, idx['leftHip']]
    knee = joint_data[f, idx['leftKnee']]
    ankle = joint_data[f, idx['leftAnkle']]
    toe = joint_data[f, idx['leftToe']]
    calc = joint_data[f, idx['leftCalc']]
    thorax = joint_data[f, idx['Middle_back']]
    midfoot = (toe + calc) / 2

    left_knee_angle[f] = 180 - calculate_angle(hip, knee, ankle)
    left_hip_angle[f] = 180 - calculate_angle(thorax, hip, knee)
    left_ankle_angle[f] = 180 - calculate_angle(knee, ankle, midfoot)
   
# Plot for knee angles
plt.figure()
plt.plot(time, right_knee_angle, 'r-', label='Right Knee', linewidth=1.5)
plt.plot(time, left_knee_angle, 'b-', label='Left Knee', linewidth=1.5)

print("\nPer-Squat Knee ROM (degrees):")
print("Squat\tStart\tEnd\tRightMin\tRightMax\tRightROM\tLeftMin\tLeftMax\tLeftROM")

for i, (start, end) in enumerate(zip(squat_starts, squat_ends), 1):
    rk = right_knee_angle[start:end]
    lk = left_knee_angle[start:end]

    rk_min, rk_max = np.min(rk), np.max(rk)
    lk_min, lk_max = np.min(lk), np.max(lk)
    rk_rom = rk_max - rk_min
    lk_rom = lk_max - lk_min

    print(f"{i}\t{start}\t{end}\t{rk_min:.1f}\t{rk_max:.1f}\t{rk_rom:.1f}\t{lk_min:.1f}\t{lk_max:.1f}\t{lk_rom:.1f}")

    t_mid = (time[start] + time[end]) / 2
    plt.text(t_mid, rk_min, f"{rk_rom:.1f}", fontsize=8, color='red', ha='center')
    plt.text(t_mid, lk_min, f"{lk_rom:.1f}", fontsize=8, color='blue', ha='center')

    plt.axvline(time[start], color='k', linestyle='--', linewidth=1)
    plt.axvline(time[end], color='k', linestyle='--', linewidth=1)

plt.xlabel("Time (s)")
plt.ylabel("Knee Angle (degrees)")
plt.title("Per-Squat ROM for Knee Angles")
plt.legend()
plt.grid(True)
plt.show()

# Plot for hip angles
plt.figure()
plt.plot(time, right_hip_angle, 'r-', label='Right Hip', linewidth=2)
plt.plot(time, left_hip_angle, 'b-', label='Left Hip', linewidth=2)

print("\nPer-Squat Hip ROM (degrees):")
print("Squat\tStart\tEnd\tRightROM\tLeftROM")

for i, (start, end) in enumerate(zip(squat_starts, squat_ends), 1):
    rh = right_hip_angle[start:end]
    lh = left_hip_angle[start:end]
    rh_rom = np.max(rh) - np.min(rh)
    lh_rom = np.max(lh) - np.min(lh)

    print(f"{i}\t{start}\t{end}\t{rh_rom:.1f}\t{lh_rom:.1f}")

    t_mid = (time[start] + time[end]) / 2
    plt.text(t_mid, np.max(rh), f"{rh_rom:.1f}", fontsize=9, color='red', ha='center')
    plt.text(t_mid, np.max(lh), f"{lh_rom:.1f}", fontsize=9, color='blue', ha='center')

    plt.axvline(time[start], color='k', linestyle='--', linewidth=1)
    plt.axvline(time[end], color='k', linestyle='--', linewidth=1)

plt.xlabel("Time (s)")
plt.ylabel("Hip Angle (degrees)")
plt.title("Hip Flexion/Extension ROM per Squat")
plt.legend()
plt.grid(True)
plt.show()

# Plot for ankle angles
plt.figure()
plt.plot(time, right_ankle_angle, 'r-', label='Right Ankle', linewidth=2)
plt.plot(time, left_ankle_angle, 'b-', label='Left Ankle', linewidth=2)

print("\nPer-Squat Ankle ROM (degrees):")
print("Squat\tStart\tEnd\tRightROM\tLeftROM")

for i, (start, end) in enumerate(zip(squat_starts, squat_ends), 1):
    ra = right_ankle_angle[start:end]
    la = left_ankle_angle[start:end]
    ra_rom = np.max(ra) - np.min(ra)
    la_rom = np.max(la) - np.min(la)

    print(f"{i}\t{start}\t{end}\t{ra_rom:.1f}\t{la_rom:.1f}")

    t_mid = (time[start] + time[end]) / 2
    plt.text(t_mid, np.max(ra), f"{ra_rom:.1f}", fontsize=9, color='red', ha='center')
    plt.text(t_mid, np.max(la), f"{la_rom:.1f}", fontsize=9, color='blue', ha='center')

    plt.axvline(time[start], color='k', linestyle='--', linewidth=1)
    plt.axvline(time[end], color='k', linestyle='--', linewidth=1)

plt.xlabel("Time (s)")
plt.ylabel("Ankle Angle (degrees)")
plt.title("Ankle Dorsiflexion ROM per Squat")
plt.legend()
plt.grid(True)
plt.show()

# Plot for spine angle
spine_angle = np.zeros(n_frames)
for f in range(n_frames):
    sacrum = joint_data[f, idx['Lower_back']]
    thorax = joint_data[f, idx['Middle_back']]
    neck = joint_data[f, idx['Upper_back']]
    spine_angle[f] = calculate_angle(sacrum, thorax, neck)

plt.figure()
plt.plot(time, spine_angle, 'k-', linewidth=2, label='Spine Angle')
plt.xlabel("Time (s)")
plt.ylabel("Spine Angle (degrees)")
plt.title("Spine Angle Over Time")
plt.grid(True)

print("\nPer-Squat Spine ROM (degrees):")
print("Squat\tStart\tEnd\tSpineROM")

for i, (start, end) in enumerate(zip(squat_starts, squat_ends), 1):
    sp = spine_angle[start:end]
    rom = np.max(sp) - np.min(sp)
    print(f"{i}\t{start}\t{end}\t{rom:.1f}")
    t_mid = (time[start] + time[end]) / 2
    plt.text(t_mid, np.max(sp), f"{rom:.1f}", fontsize=9, color='black', ha='center')
    plt.axvline(time[start], color='k', linestyle='--', linewidth=1)
    plt.axvline(time[end], color='k', linestyle='--', linewidth=1)

plt.legend()
plt.show()

# Plot for sacrum-mandible distance
mandible_index = 15
sacrum_mandible_dist = np.zeros(n_frames)
for f in range(n_frames):
    sacrum = joint_data[f, idx['Lower_back']]
    mandible = joint_data[f, mandible_index]
    sacrum_mandible_dist[f] = np.linalg.norm(mandible - sacrum)

plt.figure()
plt.plot(time, sacrum_mandible_dist, 'm-', label='Sacrum-Mandible Distance', linewidth=1.5)
plt.xlabel("Time (s)")
plt.ylabel("Distance (cm)")
plt.title("Vertical Distance: Mandible to Sacrum")
plt.grid(True)

print("\nPer-Squat Head-Pelvis Distance ROM (cm):")
print("Squat\tStart\tEnd\tMin\tMax\tROM")

for i, (start, end) in enumerate(zip(squat_starts, squat_ends), 1):
    d = sacrum_mandible_dist[start:end]
    d_min, d_max = np.min(d), np.max(d)
    rom = d_max - d_min
    print(f"{i}\t{start}\t{end}\t{d_min:.2f}\t{d_max:.2f}\t{rom:.2f}")
    t_mid = (time[start] + time[end]) / 2
    plt.text(t_mid, d_max, f"{rom:.2f}", fontsize=9, color='magenta', ha='center')
    plt.axvline(time[start], color='k', linestyle='--', linewidth=1)
    plt.axvline(time[end], color='k', linestyle='--', linewidth=1)

plt.legend()
plt.show()

# Plot for head position X-axis
head_shift_x = np.zeros(n_frames)
for f in range(n_frames):
    sacrum_x = joint_data[f, idx['Lower_back'], 0]
    mandible_x = joint_data[f, mandible_index, 0]
    head_shift_x[f] = mandible_x - sacrum_x

plt.figure()
plt.plot(time, head_shift_x, 'm-', linewidth=1.5)
plt.axhline(0, color='k', linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Mandible-Sacrum X (cm)")
plt.title("Head Position X-axis")
plt.grid(True)

print("\nPer-Squat Head Shift in X (cm):")
print("Squat\tStart\tEnd\tMin\tMax\tROM")

for i, (start, end) in enumerate(zip(squat_starts, squat_ends), 1):
    x_shift = head_shift_x[start:end]
    x_min, x_max = np.min(x_shift), np.max(x_shift)
    x_rom = x_max - x_min
    print(f"{i}\t{start}\t{end}\t{x_min:.2f}\t{x_max:.2f}\t{x_rom:.2f}")
    t_mid = (time[start] + time[end]) / 2
    plt.text(t_mid, x_max, f"{x_rom:.2f}", fontsize=9, color='magenta', ha='center')
    plt.axvline(time[start], color='k', linestyle='--', linewidth=1)
    plt.axvline(time[end], color='k', linestyle='--', linewidth=1)

plt.legend()
plt.show()