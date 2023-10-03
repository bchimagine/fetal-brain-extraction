import argparse
import logging
import os
import sys
import time

import torch
import yaml
from monai.data import decollate_batch
from monai.inferers import SliceInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from torch.optim import SGD

from data_generator_monai import FetalTrainData
from model_zoo import get_network


def train(args):
    with open(args.cfg, 'r') as file:
        configs = yaml.safe_load(file)

    if not os.path.exists(configs["save_path"]):
        os.makedirs(configs["save_path"])

    logging.basicConfig(filename=configs["save_path"] + "/log_train.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(str(configs))

    fetal_data = FetalTrainData(configs)
    train_dataloader, val_dataloader = fetal_data.load_data()

    model = get_network(configs)
    device = args.device
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    if configs["optimizer"] == "SGD":
        optimizer = SGD(model.parameters(),
                        lr=configs["learning_rate"] * 1000,
                        momentum=0.9,
                        weight_decay=0.00004,
                        )
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=configs["learning_rate"],
                                     # weight_decay=0.00004,
                                     )

    loss_function = DiceCELoss(include_background=configs["include_background"],
                               to_onehot_y=True,
                               softmax=True,
                               squared_pred=True,
                               batch=True,
                               smooth_nr=0.00001,
                               smooth_dr=0.00001,
                               lambda_dice=0.6,
                               lambda_ce=0.4,
                               )

    dice_metric = DiceMetric(include_background=configs["include_background"],
                             reduction="mean",
                             get_not_nans=False,
                             ignore_empty=False)

    img_size = configs["img_size"]
    max_epochs = configs["max_epochs"]
    val_interval = configs["val_interval"]
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=configs["classes_num"])])
    post_label = Compose([AsDiscrete(to_onehot=configs["classes_num"])])

    # Start training
    logging.info("-" * 30 + "training starts" + "-" * 30)

    step_start = time.time()
    for epoch in range(max_epochs):
        print("-" * 20)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_dataloader:
            step += 1
            inputs, labels = (batch_data["image"].to(device),
                              batch_data["label"].to(device),
                              )

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_dataloader:
                    val_inputs, val_labels = (val_data["image"].to(device),
                                              val_data["label"].to(device),
                                              )

                    infer = SliceInferer(roi_size=(img_size, img_size), sw_batch_size=1, cval=-1, spatial_dim=2,
                                         progress=False)
                    val_outputs = infer(val_inputs, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()

                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if (epoch + 1) % 10 == 0 or (epoch + 1) == max_epochs:
                    save_mode_path = os.path.join(configs["save_path"], configs["model_name"] +
                                                  '_checkpoint-%s.pth' % (epoch + 1))
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info(f"saved model at current epoch: {epoch + 1}, current mean dice: {metric:.4f}")

                elif metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    save_mode_path = os.path.join(configs["save_path"], configs["model_name"] +
                                                  '_checkpoint-%s.pth' % (epoch + 1))
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info(f"saved model at current epoch: {epoch + 1}, current best mean dice: {metric:.4f}"
                                 f" at epoch: {best_metric_epoch}")

    train_time = time.time() - step_start
    logging.info(f"train completed in {train_time:.4f} seconds "  f"best_metric: {best_metric:.4f} "
                 f"" f"at epoch: {best_metric_epoch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',
                        type=str,
                        default='../configs/train_configs.yml',
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

    train(args)
