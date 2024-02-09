# Fetal-BET: Brain Extraction Tool for Fetal MRI

Welcome to the Fetal-BET (Fetal Brain Extraction Tool) repository! 
We've developed a powerful deep learning-based solution for automatic
fetal brain extraction in MRI scans. Our method, built on a vast dataset
of 72,000 images, excels in extracting fetal brain structures from diverse
MRI sequences and scanning conditions. It's fast, accurate, and adaptable. 

![Example Segmentation Result](./plots/figures/examples.pdf)
*Sample MRI segmentation result using our methods.*

## Example Video

[Watch the video](./src/figures/combined_image_stacks.gif)

Click on the image above to watch the video.


## Table of Contents
- [Features](#features)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Docker](#docker)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Dataset](#dataset)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- Implementation of various state-of-the-art segmentation architectures.
- Docker support for reproducibility.
- Comprehensive evaluation scripts.
- Sample Jupyter notebooks for interactive experimentation. (will be added)

## Installation

### Requirements

- Python 3.8+
- Docker 20.10+ (optional but recommended)

```bash
# Clone the repository
git clone git@github.com:bchimagine/fetal-brain-extraction.git
cd fetal-brain-extraction

# Install Python dependencies
pip install -r requirements.txt
```
### Docker
```bash
# Pull the Docker Image
docker pull faghihpirayesh/fetal-bet
```

## Usage

### Training
```bash
# To train a model on your own dataset, you can use the train.py script. Here's an example command:
python train.py --dataset /path/to/dataset --model unet --epochs 50
```
### Inference
```bash
# For inference on new data, use the inference.py script. Here's an example:
python inference.py --input /path/to/input.nii.gz --output /path/to/output.nii.gz --model saved_models/attunet.pth

# Once you have pulled the image, you can run it as a Docker container. Below is an example command:
docker run -v /path/to/host/data:/path/in/container fetal-bet \
  --data_path /path/in/container/dataset/ \
  --save_path /path/in/container/prediction
```

## Dataset
In this study, our dataset comprises fetal MRI data obtained over a span of approximately 20 years from Boston Children's Hospital. These MRI acquisitions encompassed various MRI scanner types, including 1.5T GE, Philips, Siemens, and 3T Siemens scanners, specifically Skyra, Prisma, and Vida models. Ethical approval was obtained from the Institutional Review Board, and informed consent was secured from all participants for prospective fetal MRIs.

The MRI acquisition protocols included the acquisition of multiple types of images, such as T2-weighted (T2W) 2D sequences with in-plane resolutions ranging from 1 to 1.25 mm, diffusion-weighted imaging (DWI) with an in-plane resolution of 2 mm, and functional MRI (fMRI) images with an isotropic resolution of 2-3 mm.

The datasets presented in this study will be made available upon reasonable request to the corresponding author. Please direct any requests for dataset access to "razieh.faghihpirayesh@childrens.harvard.edu" via email.

## Acknowledgement
This research was supported in part by the National Institute of Biomedical Imaging and Bioengineering, the National Institute of Neurological Disorders and Stroke, and Eunice Kennedy Shriver National Institute of Child Health and Human Development of the National Institutes of Health (NIH) under award numbers R01NS106030, R01EB018988, R01EB031849, R01EB032366, and R01HD109395; and in part by the Office of the Director of the NIH under award number S10OD025111. This research was also partly supported by NVIDIA Corporation and utilized NVIDIA RTX A6000 and RTX A5000 GPUs. The content of this publication is solely the responsibility of the authors and does not necessarily represent the official views of the NIH, NSF, or NVIDIA.
