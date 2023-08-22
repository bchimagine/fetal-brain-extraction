import os
import matplotlib.pyplot as plt
from glob import glob
import monai.transforms as tr
import torch

from skimage.segmentation import slic
import numpy as np
from monai.transforms import Transform, MapTransform
from scipy.ndimage import gaussian_filter

import SimpleITK as sitk
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def generate_random_number(low, high):
    random_number = torch.rand(1) * (high - low) + low
    return random_number.item()


class CustomSuperpixelMask(Transform):
    def __init__(self, keys, intensity_option, n_segments, compactness, random_range=(-1, 1), p=1.0):
        self.img_key = keys[0]
        self.mask_key = keys[1]
        self.n_segments = np.random.randint(n_segments[0], n_segments[1])
        self.compactness = compactness
        self.random_range = random_range
        self.p = p

        self.intensity_option = intensity_option

    def __call__(self, data):
        if torch.rand(1) > self.p:
            return data

        img = data[self.img_key][0]
        mask = data[self.mask_key][0].bool()

        if mask.any():
            segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, mask=mask)
            segment_vals = np.unique(segments[segments != 0])
            selected_segment = np.random.choice(segment_vals)
            segment_mask = torch.from_numpy(segments == selected_segment)

            intensity_option = self.intensity_option
            if intensity_option == 'brighten':
                factor_map = np.random.uniform(2, 2, size=data[self.img_key][:, mask & segment_mask].shape)
                factor_map = gaussian_filter(factor_map, sigma=0.4)
                data[self.img_key][:, mask & segment_mask] *= torch.from_numpy(factor_map).to(data[self.img_key].device)

            elif intensity_option == 'blur':
                blurred_segment = gaussian_filter(img.cpu().numpy(), sigma=generate_random_number(1.5, 1.5))
                data[self.img_key][:, mask & segment_mask] = \
                    torch.from_numpy(blurred_segment).to(data[self.img_key].device)[mask & segment_mask]

            elif intensity_option == 'noise':
                random_noise = torch.normal(mean=0, std=generate_random_number(1, 1),
                                            size=((mask & segment_mask).sum().item(),)).type(data[self.img_key].dtype)
                random_noise = gaussian_filter(random_noise, sigma=1.0)

                data[self.img_key][:, mask & segment_mask] += torch.from_numpy(random_noise)

            elif intensity_option == 'darken':
                # darkening_factor = generate_random_number(0.8, 0.8)
                # data[self.img_key][:, mask & segment_mask] *= darkening_factor
                # data[self.img_key][:, mask & segment_mask] *= (darkening_factor*0.8)
                factor_map = np.random.uniform(0.3, 0.3, size=data[self.img_key][:, mask & segment_mask].shape)
                factor_map = gaussian_filter(factor_map, sigma=0.4)
                data[self.img_key][:, mask & segment_mask] *= torch.from_numpy(factor_map).to(data[self.img_key].device)

            # elif intensity_option == 'drop':
            #     data[self.img_key][:, mask & segment_mask] = 0
            #     data[self.mask_key][:, mask & segment_mask] = 0

        return data


def transform(og, intensity_option='darken'):
    if og:
        og_transforms = tr.Compose([
            tr.LoadImaged(keys=["image", "label"]),
            tr.EnsureChannelFirstd(keys=["image", "label"]),
            tr.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, -1.0), mode=("bilinear", "nearest")),
            tr.SqueezeDimd(keys=["image", "label"], dim=-1, update_meta=True),
            tr.NormalizeIntensityd(keys=["image"], subtrahend=0.0, divisor=None, nonzero=True),
            tr.ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(256, 256)),
            # tr.Resized(keys=["image", "label"], spatial_size=(256, 256)),
        ])

        return og_transforms

    else:
        transforms = tr.Compose([
            tr.LoadImaged(keys=["image", "label"]),
            tr.EnsureChannelFirstd(keys=["image", "label"]),
            tr.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, -1.0), mode=("bilinear", "nearest")),
            tr.SqueezeDimd(keys=["image", "label"], dim=-1, update_meta=True),
            tr.NormalizeIntensityd(keys=["image"], subtrahend=0.0, divisor=None, nonzero=True),
            tr.ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(256, 256)),
            # tr.Resized(keys=["image", "label"], spatial_size=(256, 256)),
            CustomSuperpixelMask(keys=["image", "label"], intensity_option=intensity_option, n_segments=[4, 5],
                                 compactness=0.1, random_range=(-1, 1),
                                 p=1)
        ])
        return transforms


train_path = '../../datasets/train_all'
train_images = sorted(glob(os.path.join(train_path, 'images', "img_*.nii.gz")), key=os.path.getmtime)
train_labels = sorted(glob(os.path.join(train_path, 'masks', "mask_*.nii.gz")), key=os.path.getmtime)

files = [{"image": image_name, "label": label_name} for
         image_name, label_name in zip(train_images, train_labels)]

og_transforms = transform(og=True)
brighten_transforms = transform(og=False, intensity_option='brighten')
blur_transforms = transform(og=False, intensity_option='blur')
noise_transforms = transform(og=False, intensity_option='noise')
darken_transforms = transform(og=False, intensity_option='darken')
i = 182
j = 19805
k = 36456
save_path = 'figures'
img = [
    og_transforms(files[i]),
    brighten_transforms(files[i]),
    blur_transforms(files[i]),
    noise_transforms(files[i]),
    darken_transforms(files[i]),

    og_transforms(files[j]),
    brighten_transforms(files[j]),
    blur_transforms(files[j]),
    noise_transforms(files[j]),
    darken_transforms(files[j]),

    og_transforms(files[k]),
    brighten_transforms(files[k]),
    blur_transforms(files[k]),
    noise_transforms(files[k]),
    darken_transforms(files[k]),
]

titles = [
    r"\textbf{Original}",
    r"\textbf{Brighten}",
    r"\textbf{Blur}",
    r"\textbf{Noise}",
    r"\textbf{Darken}"
]

fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(14, 7))

c = 0
for i, row in enumerate(axes):
    for j, cell in enumerate(row):
        img_array = img[c]
        cell.imshow(img_array["image"][0, :, :], cmap='gray', aspect='auto', origin='lower')
        # cell.text(0.05, 0.95, titles[c], color='white', ha='left', va='top', transform=cell.transAxes)
        cell.get_xaxis().set_ticks([])
        cell.get_yaxis().set_ticks([])
        if i == 0:
            cell.set_title(titles[c], fontsize=20)

        c += 1

plt.tight_layout()
plt.subplots_adjust(wspace=0.04, hspace=0.04)
plt.savefig(os.path.join(save_path, 'augmented_samples.pdf'), bbox_inches='tight', pad_inches=0)
plt.show()
