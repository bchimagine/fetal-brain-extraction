import os
import yaml
import numpy as np
import pandas as pd
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.segmentation import slic

import torch
import monai.transforms as tr
from monai.data import Dataset, CacheDataset, DataLoader, ThreadDataLoader
from monai.transforms import MapTransform, Transform
from monai.utils import first, set_determinism

import warnings

warnings.filterwarnings('ignore')


# print_config()
# Set deterministic training for reproducibility

# ____________________________________________________________________________________________________

def generate_random_number(low, high):
    random_number = torch.rand(1) * (high - low) + low
    return random_number.item()


# ____________________________________________________________________________________________________

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


# ____________________________________________________________________________________________________

class CustomSuperpixelMask(Transform):
    def __init__(self, keys, n_segments, compactness, random_range=(-1, 1), p=0.5):
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

        if mask.sum() > 500:
            segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, mask=mask)
            segment_vals = np.unique(segments[segments != 0])

            selected_segment = np.random.choice(segment_vals)
            segment_mask = torch.from_numpy(segments == selected_segment)

            intensity_option = np.random.choice(['brighten', 'blur', 'noise', 'darken'])
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
                factor_map = np.random.uniform(0.7, 0.8, size=data[self.img_key][:, mask & segment_mask].shape)
                factor_map = gaussian_filter(factor_map, sigma=0.4)
                data[self.img_key][:, mask & segment_mask] *= torch.from_numpy(factor_map).to(data[self.img_key].device)

        return data


# ____________________________________________________________________________________________________

class FetalTrainData:
    def __init__(self, config):
        self.config = config

    def transformations(self):
        train_trans = [
            tr.LoadImaged(keys=["image", "label"]),
            tr.EnsureChannelFirstd(keys=["image", "label"]),
            tr.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, -1.0), mode=("bilinear", "nearest")),
            tr.SqueezeDimd(keys=["image", "label"], dim=-1, update_meta=True),
            tr.NormalizeIntensityd(keys=["image"], subtrahend=0.0, divisor=None, nonzero=True),
            tr.Resized(keys=["image", "label"], spatial_size=(self.config["img_size"], self.config["img_size"])),
        ]

        val_trans = [
            tr.LoadImaged(keys=["image", "label"]),
            tr.EnsureChannelFirstd(keys=["image", "label"]),
            tr.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, -1.0), mode=("bilinear", "nearest")),
            SliceWiseNormalizeIntensityd(keys=["image"], subtrahend=0.0, divisor=None, nonzero=True),
            tr.Resized(keys=["image", "label"], spatial_size=(self.config["img_size"], self.config["img_size"], -1)),
        ]

        if self.config["augmentation"]:
            spatial_aug = tr.OneOf([
                tr.RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
                tr.RandRotate90d(keys=["image", "label"], spatial_axes=(0, 1), prob=0.5),
                tr.RandRotated(keys=["image", "label"], range_x=(0.2, 1.0), prob=0.6),
                tr.RandZoomd(keys=["image", "label"], min_zoom=0.8, max_zoom=1.3, prob=0.6),
                tr.RandAffined(keys=["image", "label"], padding_mode="zeros",
                               rotate_range=(np.pi / 4, np.pi / 4), shear_range=(0.5, 0.5),
                               translate_range=(30, 30), mode=("bilinear", "nearest"), prob=0.5),
            ])

            intensity_aug = tr.OneOf([
                tr.RandGaussianNoised(keys=["image"], mean=0, std=0.4, prob=0.5),
                tr.RandBiasFieldd(keys=["image"], degree=4, coeff_range=(0.05, 0.1), prob=0.6),
                tr.RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), prob=0.4),
                # CustomSuperpixelMask(keys=["image", "label"], n_segments=[2, 3], compactness=0.09, random_range=(
                # -1, 1), p=1)
            ])

            train_trans.append(spatial_aug)
            train_trans.append(intensity_aug)

        return tr.Compose(train_trans), tr.Compose(val_trans)

    def load_data(self):
        train_transforms, val_transforms = self.transformations()

        # Load train data
        if self.config["train_data_type"] == "path":
            train_images = sorted(glob(os.path.join(self.config["train_data_paths"], 'images', "img_*.nii.gz")))
            train_labels = sorted(glob(os.path.join(self.config["train_data_paths"], 'masks', "mask_*.nii.gz")))

        elif self.config["train_data_type"] == "file":
            train_images = []
            train_labels = []
            for path in self.config["train_data_paths"]:
                data = pd.read_csv(path)
                train_images += data.iloc[:, 0].tolist()
                train_labels += data.iloc[:, 1].tolist()

        train_files = [{"image": image_name, "label": label_name} for
                       image_name, label_name in zip(train_images, train_labels)]

        # Load validation data
        if self.config["validation_data_type"] == "path":
            val_images = sorted(glob(os.path.join(self.config["val_data_paths"], 'images', "img_*.nii.gz")))
            val_labels = sorted(glob(os.path.join(self.config["val_data_paths"], 'masks', "mask_*.nii.gz")))

        elif self.config["validation_data_type"] == "file":
            val_images = []
            val_labels = []
            for path in self.config["validation_data_paths"]:
                data = pd.read_csv(path)
                val_images += data.iloc[:, 0].tolist()
                val_labels += data.iloc[:, 1].tolist()

        val_files = [{"image": image_name, "label": label_name} for
                     image_name, label_name in zip(val_images, val_labels)]

        # Loader
        if self.config["fast_training"]:
            train_dataset = CacheDataset(data=train_files,
                                         transform=train_transforms,
                                         cache_rate=1.0,
                                         num_workers=8,
                                         copy_cache=False)

            train_dataloader = ThreadDataLoader(train_dataset,
                                                num_workers=0,
                                                batch_size=self.config["batch_size"],
                                                shuffle=True)

            val_dataset = CacheDataset(data=val_files,
                                       transform=val_transforms,
                                       cache_rate=1.0,
                                       num_workers=5,
                                       copy_cache=False)

            val_dataloader = ThreadDataLoader(val_dataset,
                                              num_workers=0,
                                              batch_size=1)

        else:
            train_dataset = Dataset(data=train_files,
                                    transform=train_transforms)

            train_dataloader = DataLoader(train_dataset,
                                          batch_size=self.config["batch_size"],
                                          shuffle=True,
                                          num_workers=8,
                                          pin_memory=False)

            val_dataset = Dataset(data=val_files,
                                  transform=val_transforms)

            val_dataloader = DataLoader(val_dataset,
                                        batch_size=1,
                                        num_workers=4)

        return train_dataloader, val_dataloader


# ____________________________________________________________________________________________________

class FetalTestData:
    def __init__(self, config):
        self.config = config

    def transformations(self):
        test_transforms_list = [
            tr.LoadImaged(keys=["image", "label"]),
            tr.EnsureChannelFirstd(keys=["image", "label"]),
            tr.Spacingd(keys="image", pixdim=(1.0, 1.0, -1.0), mode="bilinear"),
            SliceWiseNormalizeIntensityd(keys=["image"], subtrahend=0.0, divisor=None, nonzero=True),
            # tr.Resized(keys="image", spatial_size=(self.config["img_size"], self.config["img_size"], -1))
            # tr.ResizeWithPadOrCropd(keys="image", spatial_size=(self.config["img_size"], self.config["img_size"], -1)),
        ]
        return test_transforms_list

    def load_data(self):
        test_transforms_list = self.transformations()

        if self.config["test_data_type"] == "path":
            # test_images = sorted(glob(os.path.join(self.config["test_data_paths"], 'images', "img_*.nii.gz")))
            # test_labels = sorted(glob(os.path.join(self.config["test_data_paths"], 'masks', "mask_*.nii.gz")))

            test_images = sorted(glob(os.path.join(self.config["test_data_paths"], 'data', '**/*.nii.gz'),
                                      recursive=True))
            test_labels = sorted(glob(os.path.join(self.config["test_data_paths"], 'manual-masks', '**/*.nii.gz'),
                                      recursive=True))

        elif self.config["test_data_type"] == "file":
            test_images = []
            test_labels = []
            for path in self.config["test_data_paths"]:
                data = pd.read_csv(path)
                test_images += data.iloc[:, 0].tolist()
                test_labels += data.iloc[:, 1].tolist()

        test_files = [{"image": image_name, "label": label_name} for
                      image_name, label_name in zip(test_images, test_labels)]

        test_dataset = Dataset(data=test_files, transform=tr.Compose(test_transforms_list))
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=1,
                                     num_workers=0)

        return test_dataloader, test_files, test_transforms_list