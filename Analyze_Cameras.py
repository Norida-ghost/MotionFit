import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from scipy.signal import find_peaks
from scipy.stats import pearsonr

# Weighting for ranking  (should sum to 1)
WEIGHTS = {
    'Squats': 0.3, # Squats detected
    'MissingValues': 0.2, # Missing or incomplete data
    'NoiseLevel': 0.2, # Noise level in data (Should be low)
    'RMSE_vs_Ideal': 0.2, # Root Mean Square Error vs Ideal 
    'Correlation_vs_Ideal': 0.1 # How is the alignment with the ideal
}

# Utilities 
def calculate_y_signal(data, joint_index=18):
    n_frames = len(data)
    y = np.zeros(n_frames)
    try:
        y = data[f'Y{joint_index}'].values
    except:
        pass
    return y

def detect_squats(y_ear, frame_rate=100, prominence=0.03):
    inv_y = -y_ear
    bottoms, _ = find_peaks(inv_y, distance=int(0.05 * frame_rate), prominence=prominence)
    return bottoms

def compare_to_ideal(y_target, y_ideal):
    n = min(len(y_target), len(y_ideal))
    y_target = y_target[:n]
    y_ideal = y_ideal[:n]
    rmse = np.sqrt(np.mean((y_target - y_ideal) ** 2))
    corr, _ = pearsonr(y_target, y_ideal)
    return rmse, corr

def normalize(series, higher_is_better=True):
    s = series.astype(float)
    if s.max() == s.min():
        return pd.Series(1.0, index=s.index)
    norm = (s - s.min()) / (s.max() - s.min())
    return norm if higher_is_better else 1 - norm

# File selection
Tk().withdraw()
ideal_file = filedialog.askopenfilename(title="Select the IDEAL CSV file", filetypes=[("CSV files", "*.csv")])
if not ideal_file:
    print("Ideal file not selected."); exit()

file_paths = filedialog.askopenfilenames(title="Select CSV files to compare", filetypes=[("CSV files", "*.csv")])
if not file_paths:
    print("No comparison files selected."); exit()

# Load Ideal file 
print(f"Loading ideal file: {ideal_file}")
ideal_data = pd.read_csv(ideal_file)
ideal_y = calculate_y_signal(ideal_data)

# Compare each file 
results = []
for path in file_paths:
    print(f"Analyzing: {os.path.basename(path)}")
    try:
        data = pd.read_csv(path)
        y = calculate_y_signal(data)
        n_frames = len(data)

        squat_count = len(detect_squats(y, frame_rate=100, prominence=0.03))
        missing = data.isna().sum().sum()
        noise = np.std(np.diff(y))
        rmse, corr = compare_to_ideal(y, ideal_y)

        results.append({
            'File': os.path.basename(path),
            'Squats': squat_count,
            'MissingValues': missing,
            'NoiseLevel': noise,
            'RMSE_vs_Ideal': rmse,
            'Correlation_vs_Ideal': corr
        })

    except Exception as e:
        print(f"Error with {path}: {e}")
        results.append({
            'File': os.path.basename(path),
            'Squats': 0,
            'MissingValues': -1,
            'NoiseLevel': -1,
            'RMSE_vs_Ideal': -1,
            'Correlation_vs_Ideal': -1
        })

# Create DataFrame
df = pd.DataFrame(results)

# === Ranking: normalize and combine
norm_squats = normalize(df['Squats'], True)
norm_missing = normalize(df['MissingValues'], False)
norm_noise = normalize(df['NoiseLevel'], False)
norm_rmse = normalize(df['RMSE_vs_Ideal'], False)
norm_corr = normalize(df['Correlation_vs_Ideal'], True)

df['RankScore'] = (
    norm_squats * WEIGHTS['Squats'] +
    norm_missing * WEIGHTS['MissingValues'] +
    norm_noise * WEIGHTS['NoiseLevel'] +
    norm_rmse * WEIGHTS['RMSE_vs_Ideal'] +
    norm_corr * WEIGHTS['Correlation_vs_Ideal']
)

# Sort and output
df_sorted = df.sort_values(by='RankScore', ascending=False)

print("\n=== Ranked Summary ===")
print(df_sorted[['File', 'Squats', 'MissingValues', 'NoiseLevel', 'RMSE_vs_Ideal', 'Correlation_vs_Ideal', 'RankScore']].to_string(index=False))

# Ask for output folder
output_dir = filedialog.askdirectory(title="Select folder to save the ranked summary")
if not output_dir:
    print("No output folder selected."); exit()

# Create output path
output_csv = os.path.join(output_dir, "ranked_summary.csv")
df_sorted.to_csv(output_csv, index=False)
print(f"\nSaved ranked summary to: {output_csv}")

print("\n Top 5 files:")
print(df_sorted.head(5)[['File', 'RankScore']].to_string(index=False))

# === Plot: Top scores
plt.figure(figsize=(12, 6))
top = df_sorted.head(10)
plt.bar(top['File'], top['RankScore'], color='purple')
plt.xticks(rotation=90)
plt.ylabel("Composite Rank Score")
plt.title("Top 10 Ranked CSV Files")
plt.grid(True)
plt.tight_layout()
plt.show()
