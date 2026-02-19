# tested for Python 3.9.23 
# It is recommended to create a conda environment first, e.g.:
# conda create --name ploXt python=3.9 -y
# conda activate ploXt

# 1. Install PyTorch separately to ensure correct GPU/CUDA version.
#    This command is for CUDA 11.7.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126


# 3. Install MMDetection in editable mode.
#    This assumes the 'mmdetection' directory is in the project root.
uv pip install -e mmdetection --no-build-isolation


# 2. Install all other packages from the consolidated requirements file.
#    `openmim` is listed in requirements.txt and will be used by `pip`
#    to correctly install `mmcv-full`.
uv pip install -r requirements.txt



