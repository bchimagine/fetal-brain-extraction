# Fetal-BET: Brain Extraction Tool for Fetal MRI

This repository contains our efforts on MRI segmentation using various neural network architectures
like U-Net, Dynamic U-Net, and Attention U-Net. The project aims to provide a comparison between these
architectures in terms of their segmentation accuracy, robustness, and efficiency.

![Example Segmentation Result](./plots/figures/examples.pdf)
*Sample MRI segmentation result using our methods.*

## Table of Contents
- [Features](#features)
- [Installation](#installation)
  - [Docker](#docker)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- Implementation of various state-of-the-art segmentation architectures.
- Docker support for reproducibility.
- Comprehensive evaluation scripts.
- Sample Jupyter notebooks for interactive experimentation.

## Installation

### Requirements

- Python 3.8+
- Docker 20.10+ (optional but recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/mri-segmentation.git
cd mri-segmentation

# Install Python dependencies
pip install -r requirements.txt
