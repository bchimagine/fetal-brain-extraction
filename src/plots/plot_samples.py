import os

import SimpleITK as sitk
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

save_path = 'figures'

paths = [
    "../datasets/t2mri_dataset/data/t2/FCB050s1/fetus_16.nii.gz",
    "../datasets/dmri_dataset/data/B1/FCB080s1_21_vol_0012.nii.gz",
    "../datasets/fmri_dataset/dMRIs3D/f1070s2/rs-fMRI_120_time_pts_10/vol_0010.nii.gz",
]

titles = [
    r"T2-weighted",
    r"diffusion-weighted",
    r"functional MRI"
]

fig, axes = plt.subplots(nrows=3, ncols=3, gridspec_kw={'height_ratios': [4, 1, 1]}, figsize=(24, 12),
                         constrained_layout=True)

for i in range(axes.shape[1]):
    img = sitk.ReadImage(paths[i])
    img_array = sitk.GetArrayFromImage(img)

    axes[0, i].imshow(img_array[img_array.shape[0] // 2, :, :], cmap='gray', aspect='auto')
    axes[1, i].imshow(img_array[:, img_array.shape[1] // 2, :], cmap='gray', aspect='auto')
    axes[2, i].imshow(img_array[:, :, img_array.shape[2] // 2], cmap='gray', aspect='auto')

    axes[0, i].set_title(r'\textbf{\fontsize{' + str(40) + '}{0}\selectfont ' + titles[i] + '}')

    for j in range(axes.shape[0]):
        axes[j, i].get_xaxis().set_ticks([])
        axes[j, i].get_yaxis().set_ticks([])
        axes[j, i].axis('off')

plt.tight_layout()
plt.subplots_adjust(wspace=0.04, hspace=0.04)
plt.savefig(os.path.join(save_path, 'inutero.pdf'), bbox_inches='tight', pad_inches=0)
plt.show()
