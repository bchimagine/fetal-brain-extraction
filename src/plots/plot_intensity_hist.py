import glob

import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def compute_average(nii_path):
    volume = nib.load(nii_path).get_fdata()
    (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    return np.mean(volume)


data = pd.DataFrame()
path = "../../datasets/t2mri_dataset/data_csv/*.csv"
for fname in glob.glob(path):
    data_ = pd.read_csv(fname, on_bad_lines='skip')
    data = pd.concat([data, data_])
T2_paths = data['image']

data = pd.DataFrame()
path = "../../datasets/dmri_dataset/data_csv/*.csv"
for fname in glob.glob(path):
    data_ = pd.read_csv(fname, on_bad_lines='skip')
    data = pd.concat([data, data_])
dwi_paths = data['image']

data = pd.DataFrame()
path = "../../datasets/fmri_dataset/data_csv/*.csv"
for fname in glob.glob(path):
    data_ = pd.read_csv(fname, on_bad_lines='skip')
    data = pd.concat([data, data_])
fmri_paths = data['image']

T2_averages = [compute_average(path) for path in T2_paths]
dwi_averages = [compute_average(path) for path in dwi_paths]
fmri_averages = [compute_average(path) for path in fmri_paths]

min_grey = 0#min(min(T2_averages), min(dwi_averages), min(fmri_averages))
max_grey = 101#max(max(T2_averages), max(dwi_averages), max(fmri_averages))

# Calculate bin size (for example, every 5 units)
bin_size = 5

# Generate bins from the minimum to the maximum grey level in steps of bin_size
bins = [i for i in range(int(min_grey), int(max_grey) + bin_size, bin_size)]
# Create a new figure and axis object
fig, ax = plt.subplots()

# Plot histograms with muted colors
ax.hist(T2_averages, bins, alpha=0.5, label='T2', color='steelblue', log=False)
ax.hist(dwi_averages, bins, alpha=0.5, label='DWI', color='firebrick', log=False)
ax.hist(fmri_averages, bins, alpha=0.5, label='fMRI', color='forestgreen', log=False)

# Adding grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Use LaTeX for labels and title
ax.set_xlabel(r'Average Grey Level')
ax.set_ylabel(r'Count')
ax.set_title(r'Histogram of Average Grey Levels for Different Modalities')

ax.legend(loc='upper right')

# Display the plot
plt.tight_layout()
plt.show()
