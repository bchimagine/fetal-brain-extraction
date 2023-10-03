from monai.networks.layers import Norm
from monai.networks.nets import UNet, DynUNet, AttentionUnet, SwinUNETR, BasicUNet

UNet = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(64, 128, 256, 512, 1024),#(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    # num_res_units=2,
    norm=Norm.BATCH,
    kernel_size=3,
    up_kernel_size=3,
    # act=Act.PRELU,
    dropout=0.2,
    bias=True,
)

BasicUNet = BasicUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    features=(64, 128, 256, 512, 1024, 64), #(32, 64, 128, 256, 512)
    norm=Norm.BATCH,
    dropout=0.15,
)

AttUNet = AttentionUnet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(64, 128, 256, 512, 1024),  # (32, 64, 128, 256, 512),(64, 128, 256, 512, 1024)
    strides=(2, 2, 2, 2),
    kernel_size=3,
    up_kernel_size=3,
    dropout=0.15,
)


def dynunet():
    patch_size = [256, 256, 1]
    spacing = (1.0, 1.0, -1.0)
    spacings = spacing[:2]
    sizes = patch_size[:2]
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])

    # initialise the network
    net = DynUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        # deep_supervision=True,
        # deep_supr_num=2,
        dropout=0.15,
        res_block=False,
    )
    return net


def get_network(config):
    model_name = config["model_name"]
    model_mapping = {
        "UNet": UNet,
        "AttUNet": AttUNet,
        "BasicUNet": BasicUNet,
        "DynUNet": dynunet(),
    }
    if model_name not in model_mapping:
        raise ValueError(f"Invalid model name: {model_name}")

    model = model_mapping[model_name]
    return model
