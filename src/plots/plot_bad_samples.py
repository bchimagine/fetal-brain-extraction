import os

import SimpleITK as sitk
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

save_path = 'figures'

paths = [
    "../datasets/t2mri_dataset/data/abnormality/vogm_006s1/fetus_21.nii.gz",
    "../datasets/t2mri_dataset/data/artifacts/f1212s1/f1212s1_fetus_06.nii.gz",
    "../datasets/t2mri_dataset/data/t2/FCB050s1/fetus_16.nii.gz",
    "../datasets/t2mri_dataset/data/twins/f0663s1/fetus_B_20.nii.gz",
    "../datasets/t2mri_dataset/data/abnormality/lvm0090s1/fetus_5.nii.gz",
    "../datasets/fmri_dataset/dMRIs3D/f1114s1/rs-fMRI_120_time_pts_16/vol_0010.nii.gz",
    "../datasets/fmri_dataset/dMRIs3D/f1200s2/rs-fMRI_80_time_pts_39/vol_0010.nii.gz",
    "../datasets/dmri_dataset/data/B0/0983s1_16_vol_0000.nii.gz",
    "../datasets/dmri_dataset/data/B1/0983s1_22_vol_0001.nii.gz",
    "../datasets/dmri_dataset/data/B1/1133s1_10_vol_0003.nii.gz"
]

idx = [15, 13, 14, 2, 19, 16, 15, 10, 13, 16]

titles = [
    r"\textbf{T2-weighted}",
    r"\textbf{T2-weighted}",
    r"\textbf{T2-weighted}",
    r"\textbf{T2-weighted}",
    r"\textbf{T2-weighted}",
    r"\textbf{diffusion-weighted}",
    r"\textbf{diffusion-weighted}",
    r"\textbf{diffusion-weighted}",
    r"\textbf{functional MRI}",
    r"\textbf{functional MRI}"
]

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 5))

c = 0
for i, row in enumerate(axes):
    for j, cell in enumerate(row):
        img = sitk.ReadImage(paths[c])
        img_array = sitk.GetArrayFromImage(img)

        cell.imshow(img_array[idx[c], :, :], cmap='gray', aspect='auto')
        cell.text(0.05, 0.95, titles[c], color='white', ha='left', va='top', transform=cell.transAxes)
        cell.get_xaxis().set_ticks([])
        cell.get_yaxis().set_ticks([])
        # cell.set_title(labels[c])
        # cell.axis('off')

        c += 1

plt.tight_layout()
plt.subplots_adjust(wspace=0.04, hspace=0.04)
plt.savefig(os.path.join(save_path, 'bad_samples.pdf'), bbox_inches='tight', pad_inches=0)
plt.show()
