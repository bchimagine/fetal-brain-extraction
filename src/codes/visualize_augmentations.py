import os
from glob import glob
import random
import numpy as np
import pandas as pd
import monai.transforms as tr
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, ThreadDataLoader, TestTimeAugmentation
import torch
import matplotlib
import matplotlib.pyplot as plt
from monai.utils import first
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import numpy as np
from monai.transforms import Transform, MapTransform
from scipy.ndimage import gaussian_filter


class SliceWiseNormalizeIntensityd(MapTransform):
    def __init__(self, keys, subtrahend=0.0, divisor=None, nonzero=True):
        super().__init__(keys)
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            for i in range(image.shape[-1]):
                slice_ = image[..., i]
                if self.nonzero:
                    mask = slice_ > 0
                    if np.any(mask):
                        if self.subtrahend is None:
                            slice_[mask] = slice_[mask] - slice_[mask].mean()
                        else:
                            slice_[mask] = slice_[mask] - self.subtrahend

                        if self.divisor is None:
                            slice_[mask] /= slice_[mask].std()
                        else:
                            slice_[mask] /= self.divisor

                else:
                    if self.subtrahend is None:
                        slice_ = slice_ - slice_.mean()
                    else:
                        slice_ = slice_ - self.subtrahend

                    if self.divisor is None:
                        slice_ /= slice_.std()
                    else:
                        slice_ /= self.divisor

                image[..., i] = slice_
            d[key] = image
        return d


def generate_random_number(low, high):
    random_number = torch.rand(1) * (high - low) + low
    return random_number.item()


class CustomSuperpixelMask(Transform):
    def __init__(self, keys, n_segments, compactness, random_range=(-1, 1), p=1.0):
        self.img_key = keys[0]
        self.mask_key = keys[1]
        self.n_segments = np.random.randint(n_segments[0], n_segments[1])
        self.compactness = compactness
        self.random_range = random_range
        self.p = p

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

            intensity_option = np.random.choice(['brighten', 'blur', 'noise', 'darken'])
            print(intensity_option)
            if intensity_option == 'brighten':
                factor_map = np.random.uniform(1.05, 1.2, size=data[self.img_key][:, mask & segment_mask].shape)
                factor_map = gaussian_filter(factor_map, sigma=0.4)
                data[self.img_key][:, mask & segment_mask] *= torch.from_numpy(factor_map).to(data[self.img_key].device)

            elif intensity_option == 'blur':
                blurred_segment = gaussian_filter(img.cpu().numpy(), sigma=generate_random_number(0.9, 1.0))
                data[self.img_key][:, mask & segment_mask] = \
                    torch.from_numpy(blurred_segment).to(data[self.img_key].device)[mask & segment_mask]

            elif intensity_option == 'noise':
                random_noise = torch.normal(mean=0, std=generate_random_number(0.2, 0.4),
                                            size=((mask & segment_mask).sum().item(),)).type(data[self.img_key].dtype)
                random_noise = gaussian_filter(random_noise, sigma=1.0)

                data[self.img_key][:, mask & segment_mask] += torch.from_numpy(random_noise)

            elif intensity_option == 'darken':
                # darkening_factor = generate_random_number(0.8, 0.8)
                # data[self.img_key][:, mask & segment_mask] *= darkening_factor
                # data[self.img_key][:, mask & segment_mask] *= (darkening_factor*0.8)
                factor_map = np.random.uniform(0.7, 0.8, size=data[self.img_key][:, mask & segment_mask].shape)
                factor_map = gaussian_filter(factor_map, sigma=0.4)
                data[self.img_key][:, mask & segment_mask] *= torch.from_numpy(factor_map).to(data[self.img_key].device)

            # elif intensity_option == 'drop':
            #     data[self.img_key][:, mask & segment_mask] = 0
            #     data[self.mask_key][:, mask & segment_mask] = 0

        return data


transforms = tr.Compose([
    tr.LoadImaged(keys=["image", "label"]),
    tr.EnsureChannelFirstd(keys=["image", "label"]),
    tr.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, -1.0), mode=("bilinear", "nearest")),
    # tr.SqueezeDimd(keys=["image", "label"], dim=-1, update_meta=True),
    SliceWiseNormalizeIntensityd(keys=["image"], subtrahend=0.0, divisor=None, nonzero=True),
    tr.ResizeWithPadOrCropd(keys=["image", "label"],
                            spatial_size=(256, 256, -1)),

    # tr.RandAdjustContrastd(keys=["image"], prob=1, gamma=(0.9, 0.9)),
    # tr.RandGaussianNoised(keys=["image"], mean=0, std=0.4, prob=0.5),
    # tr.RandBiasFieldd(keys=["image"], degree=4, coeff_range=(0.05, 0.1), prob=0.6),
    tr.RandGaussianSmoothd(keys=["image"], sigma_x=(0.6, 0.6), sigma_y=(0.6, 0.6), prob=1),
    CustomSuperpixelMask(keys=["image", "label"], n_segments=[2, 3], compactness=0.09, random_range=(-1, 1), p=1)
])

# train_path = '../datasets/train_all'
# train_images = glob(os.path.join(train_path, 'images', "img_*.nii.gz"))
# train_labels = glob(os.path.join(train_path, 'masks', "mask_*.nii.gz"))
#
# files = [{"image": image_name, "label": label_name} for
#          image_name, label_name in zip(train_images[20000:], train_labels[20000:])]

data = pd.read_csv('../../datasets/dmri_dataset/data_csv/test_data.csv')
train_images = data.iloc[:, 0].tolist()
train_labels = data.iloc[:, 1].tolist()
files = [{"image": image_name, "label": label_name} for
         image_name, label_name in zip(train_images, train_labels)]


batch = 9
s = 0
cmap_mask = matplotlib.colors.ListedColormap(['none', 'red'])

org_tr = tr.Compose([
    tr.LoadImaged(keys=["image", "label"]),
    tr.EnsureChannelFirstd(keys=["image", "label"]),
    tr.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, -1.0), mode=("bilinear", "nearest")),
    # tr.SqueezeDimd(keys=["image", "label"], dim=-1, update_meta=True),
    SliceWiseNormalizeIntensityd(keys=["image"], subtrahend=0.0, divisor=None, nonzero=True),
    tr.ResizeWithPadOrCropd(keys=["image", "label"],
                            spatial_size=(256, 256, -1)),
])


inputt = org_tr(files[72])

plt.imshow(inputt["image"][0, :, :, 10], cmap="gray")
plt.tick_params(left=False, right=False, labelleft=False,
                labelbottom=False, bottom=False)

# plt.savefig("../plots/orgw.png", bbox_inches='tight')
plt.show()

# for i in range(6):
out = transforms(files[72])
plt.imshow(out["image"][0, :, :, 10], cmap="gray")
plt.imshow(out["label"][0, :, :, 10], alpha=0.0, cmap=cmap_mask)
plt.tick_params(left=False, right=False, labelleft=False,
                labelbottom=False, bottom=False)
# plt.savefig(str(i)+"t2.png", bbox_inches='tight')
plt.show()
