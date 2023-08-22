import os
import math
import time

import pandas as pd
import yaml
import argparse
import matplotlib
import numpy as np

import torch

import monai.transforms as tr
from matplotlib import pyplot as plt
from monai.data import decollate_batch, TestTimeAugmentation
from monai.handlers import from_engine
from monai.inferers import SliceInferer
from monai.metrics import HausdorffDistanceMetric, MeanIoU, DiceMetric, ConfusionMatrixMetric, \
    compute_confusion_matrix_metric
from monai.utils import first, set_determinism

from data_generator_monai import FetalTestData
from model_zoo import get_network


def plot_images(images, masks, gt=None,
                volume_dice=None, mean_slice_dice=None, slice_dice=None,
                save_name=None, display=None):
    slice_num = images.shape[-1]
    # Calculate the number of columns and rows based on the number of images
    n_cols = int(math.ceil(math.sqrt(slice_num)))
    n_rows = int(math.ceil(slice_num / n_cols))

    cmap_mask = matplotlib.colors.ListedColormap(['none', 'red'])
    cmap_gt = matplotlib.colors.ListedColormap(['none', 'blue'])

    # Create a grid of subplots with the calculated number of columns and rows
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    for i in range(slice_num):
        # Calculate the row and column indices for the current subplot
        row = i // n_cols
        col = i % n_cols

        # Plot the image with the mask overlay
        axs[row, col].imshow(images[:, :, i], cmap='gray')
        axs[row, col].imshow(masks[:, :, i], alpha=0.6, cmap=cmap_mask)

        if not (gt is None):
            axs[row, col].imshow(gt[:, :, i], alpha=0.3, cmap=cmap_gt)

        if not (slice_dice is None):
            axs[row, col].set_title(f"dice= {slice_dice[i]:.2f}")

        axs[row, col].axis('off')

    # Remove any unused subplots
    for i in range(len(images), n_rows * n_cols):
        axs.flatten()[i].set_visible(False)

    if volume_dice and mean_slice_dice:
        fig.suptitle(
            f"red: predicted, blue: manual, volume_dice= {volume_dice:.2f}, mean_slice_dice= {mean_slice_dice:.2f}")

    elif volume_dice:
        fig.suptitle(
            f"red: predicted, blue: manual, volume_dice= {volume_dice:.2f}")

    if save_name:
        plt.savefig(save_name)

    if display:
        plt.show()


def test(args):
    with open(args.cfg, 'r') as file:
        configs = yaml.safe_load(file)

    if not os.path.exists(configs["save_path"]):
        os.makedirs(configs["save_path"])

    device = args.device
    model = get_network(configs)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    model.load_state_dict(torch.load(configs["saved_model_path"], map_location=device))
    model.eval()

    fetal_test_data = FetalTestData(configs)
    test_dataloader, test_files, test_org_transforms_list = fetal_test_data.load_data()

    inferer = SliceInferer(
        roi_size=(configs["img_size"], configs["img_size"]),
        spatial_dim=2,
        progress=False
    )

    dice_metric = DiceMetric(
        include_background=configs["include_background"],
        reduction="mean",
        ignore_empty=False
    )
    iou_metric = MeanIoU(
        include_background=configs["include_background"],
        reduction="mean",
        ignore_empty=False
    )

    test_time = []
    subj_list = []
    mask_list = []
    data_type = []
    if configs["test_augmentation"]:
        # not complete yet!
        spatial_transforms = [[tr.RandFlipd(keys="image", spatial_axis=1, prob=0.8)],
                              [tr.RandRotate90d(keys="image", spatial_axes=(0, 1), prob=0.6)],
                              [tr.RandRotated(keys="image", range_x=(0.2, 1.0), prob=0.5)],
                              [tr.RandZoomd(keys="image", min_zoom=0.95, max_zoom=1.2, prob=0.6)], ]

        with torch.no_grad():
            for file in test_files:
                tta_out = []
                for trans in spatial_transforms:
                    tta_transforms = tr.Compose(test_org_transforms_list + trans)
                    tta_post_transforms = tr.Compose([tr.Activations(softmax=True),
                                                      tr.AsDiscrete(argmax=True, to_onehot=None),
                                                      ])

                    tta = TestTimeAugmentation(transform=tta_transforms,
                                               batch_size=1,
                                               num_workers=0,
                                               inferrer_fn=lambda x: inferer(x, model),
                                               device=device,
                                               orig_key="image",
                                               post_func=lambda x: tta_post_transforms(x),
                                               progress=True,
                                               return_full_data=True)

                    _out = tta(file, num_examples=5)
                    tta_out.append(_out)

                tta_output = torch.vstack(tta_out)
                tta_output_mean = torch.mean(tta_output, dim=0)

                unmodified_data = tr.LoadImaged(keys=["image", "label"])(file)
                dice = dice_metric(y_pred=tta_output_mean[None, ...], y=unmodified_data["label"][None, None, ...])
                iou = iou_metric(y_pred=tta_output_mean[None, ...], y=unmodified_data["label"][None, None, ...])

    else:
        post_transforms_list = [
            tr.Invertd(
                keys="pred",
                transform=tr.Compose(test_org_transforms_list),
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            tr.Activationsd(keys="pred", softmax=True),
            tr.AsDiscreted(keys="pred", argmax=True, to_onehot=None),
            # tr.KeepLargestConnectedComponentd(keys="pred", applied_labels=1, num_components=2, is_onehot=False),
            # tr.RemoveSmallObjectsd(keys="pred", min_size=50, connectivity=1),
        ]

        if configs["save_predictions"]:
            post_transforms_list.append(
                tr.SaveImaged(
                    keys="pred",
                    meta_keys="pred_meta_dict",
                    output_dir=configs["save_path"],
                    separate_folder=False,
                    data_root_dir=configs["data_root_dir"],
                    output_postfix="pred",
                    resample=True
                )
            )

        post_transforms = tr.Compose(post_transforms_list)

        with torch.no_grad():
            for i, test_data in enumerate(test_dataloader):
                test_inputs, test_labels = test_data["image"].to(device), test_data["label"].to(device)

                start_time = time.time()

                test_data["pred"] = inferer(test_inputs, model)
                test_data = [post_transforms(i) for i in decollate_batch(test_data)]

                test_time.append((time.time() - start_time) / test_inputs.shape[-1])

                dice = dice_metric(y_pred=from_engine(["pred"])(test_data), y=test_labels)
                iou = iou_metric(y_pred=from_engine(["pred"])(test_data), y=test_labels)

                if configs["modality"] == "T2W" or configs["modality"] == "otherscanners":
                    subject_id = os.path.basename(os.path.dirname(test_inputs.meta["filename_or_obj"][0]))
                    if subject_id.endswith('s1') or subject_id.endswith('s2'):
                        subject_id = subject_id[:-2]
                    subj_list.append(subject_id)
                    data_type.append(os.path.basename(os.path.dirname(os.path.dirname(
                        test_inputs.meta["filename_or_obj"][0]))))

                elif configs["modality"] == "DWI":
                    subject_id = os.path.basename(test_inputs.meta["filename_or_obj"][0]).split('_')[0]
                    if subject_id.endswith('s1') or subject_id.endswith('s2'):
                        subject_id = subject_id[:-2]
                    subj_list.append(subject_id)
                    data_type.append(os.path.basename(os.path.dirname(test_inputs.meta["filename_or_obj"][0])))

                elif configs["modality"] == "fMRI":
                    subject_id = os.path.dirname(test_inputs.meta["filename_or_obj"][0]).split('/')[-2][:-2]
                    subj_list.append(subject_id)
                    data_type.append('fMRI')

                if configs["plot_results"]:
                    if dice.item() < 0.90:
                        test_output = from_engine(["pred"])(test_data)
                        original_image = tr.LoadImage()(test_output[0].meta["filename_or_obj"])[0]
                        original_label = tr.LoadImage()(test_labels[0].meta["filename_or_obj"])[0]
                        plot_images(
                            original_image,
                            test_output[0].detach().cpu()[0],
                            gt=original_label,
                            volume_dice=dice.item(),
                            mean_slice_dice=None,
                            slice_dice=None,
                            save_name=None,
                            display=False
                        )
                    print(test_output[0].meta["filename_or_obj"])
                    print(dice.item())
                    print(iou.item())

    if configs["save_metrics"]:
        header = ["Method", "Modality", "Type", "Subject", "Dice", "IoU"]
        dice_list = (dice_metric.get_buffer().detach().cpu().numpy()[:, 0]).tolist()
        iou_list = (iou_metric.get_buffer().detach().cpu().numpy()[:, 0]).tolist()
        modality = [configs["modality"]] * len(dice_list)
        method = [os.path.splitext(os.path.basename(configs["saved_model_path"]))[0]] * len(dice_list)

        data = list(zip(method, modality, data_type, subj_list, dice_list, iou_list))

        file_path = os.path.join(args.save_path, configs["modality"] + "_" + method[0] + ".csv")
        df = pd.DataFrame(data, columns=header)
        df.to_csv(file_path, index=False)

        print("evaluation metric dice mean:", np.mean(dice))
        print("evaluation metric dice std:", np.std(dice))

        print("evaluation metric iou mean:", np.mean(iou_list))
        print("evaluation metric iou std:", np.std(iou_list))

    print(np.mean(test_time[1:]))
    print("evaluation metric dice:", dice_metric.aggregate())
    print("evaluation metric iou:", iou.aggregate())

    dice_metric.reset()
    iou_metric.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',
                        type=str,
                        default='../configs/test_configs.yml',
                        help='path to config file')

    parser.add_argument('--n_gpu',
                        type=int,
                        default=2,
                        help='total gpu')

    parser.add_argument('--deterministic',
                        type=int,
                        default=1,
                        help='whether use deterministic training')

    parser.add_argument('--seed',
                        type=int,
                        default=1234,
                        help='random seed')

    parser.add_argument('--device',
                        type=str,
                        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help='what device to use')

    args = parser.parse_args()

    if args.deterministic:
        set_determinism(seed=12345)
    else:
        set_determinism(seed=None)

    test(args)
