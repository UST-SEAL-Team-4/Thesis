# Thesis

Repository for all the experimental work for the thesis project

This repository will be temporary, the final output will be pushed to another repository once everything has been finalized

## Files Description

Below are the main files used for the experimental work for the thesis project
- `ISARPN.ipynb`: training notebook for the `ISARPN` module
- `ISARPN_ISAVIT.ipynb`: training notebook for `ISAViT` using `ISARPN` as the region proposal network
- `ISARPN_VIT.ipynb`: training notebook for `ViT` using `ISARPN` as the region proposal network
- `PBBOX_ISAVIT.ipynb`: training notebook for `ISAViT` using the ground truth bounding boxes as the RPN
- `PBBOX_VIT.ipynb`: training notebook for `ViT` using the ground truth bounding boxes as the RPN
- `requirements.txt`: list of the specific versions used for each module
- `RPN.ipynb`: training notebook for the `RPN` module
- `rpn_error_analysis.ipynb`: a notebook for exploring the errors presented by both `ISARPN` and `RPN` module
- `RPN_ISAVIT.ipynb`: training notebook for `ISAViT` using `RPN` as the region proposal network
- `RPN_VIT.ipynb`: training notebook for `ViT` using `RPN` as the region proposal network
- `system_demo.ipynb`: Notebook for demonstrating the entire flow of the model from input to segmented output

These folders hold more or less the same purpose as given below:
- `best_fold_weights\`: folder containing the saved weights of the best folds from the experiments
- `notebooks\`: all the outdated, obsolete, and archived jupyter notebooks used for experimentation
- `project\`: an entire folder containing python scripts created by the researchers that serve as reusable modules needed for the numerous experiments
    - Modules include dataloaders, models, preprocessing, and metrics that are needed for experiments regardless of configuration
- `statistical-treatment\`: folder for statistical treatment of the results of each experiment

## Installation

Run the command for installing the required modules
```
pip install jupyterlab numpy pandas seaborn matplotlib scikit-learn albumentations opencv-python nibabel
```

For pytorch, there is a specific installation required for cuda integration. It is recommended to check out [the official pytorch website](https://pytorch.org) for their instructions

If there is no need for gpu integration, installing pytorch is as simple as
```
pip install torch torchvision torchaudio
```

## Setup

Follow each chapter chronologically:

### Dataset installation

1. Register an account in the [grand-challenge.org](https://grand-challenge.org) website
2. Once signed in, install the dataset [here](https://valdo.grand-challenge.org/Data/). Choose the task 2 dataset.

The directory setup must be as follows
```
Main Folder
|
|____ This repository
|
|
|____ VALDO Dataset
```

### Using SynthStrip Docker for skull-stripping MRI scans

For preprocessing, there is a need to skull-strip the MRI data, follow the instructions below:

1. Install Docker
2. Run the following command in your terminal to download SynthStrip:
   `docker pull freesurfer/synthstrip`
3. Verify the image in the terminal:
   `docker images`
4. Add Docker to your PATH environment variable
5. Navigate to `notebooks/` and open the file `skull_stripping_process.ipynb`
6. Update the `DOCKER_PATH` according to the location of your `docker.exe`
7. Ensure that `SOURCE_DIR` in the notebook points to the correct location of your VALDO Dataset
8. Run the Jupyter notebook

### Creating `targets.csv`

`targets.csv` serves as the basis for which will be the target slice, normalization of pixel values to [0, 1], and where to find each file. Assuming the previous chapters for setup have already been performed properly, simply run `targets_with_stripped.ipynb` to generate the metadata
