# Get started

## Installation

This codebase is modified from the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). To support the [UMFC](https://arxiv.org/abs/2411.06921) work, we made adjustments to the original Dassl.pytorch code to set up the environment required for UMFC.

```bash
# Clone this repo
git clone https://github.com/GIT-LJc/Dassl.change.git
cd Dassl.change/

# Create a conda environment
conda create -y -n dassl python=3.8

# Activate the environment
conda activate dassl

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```
