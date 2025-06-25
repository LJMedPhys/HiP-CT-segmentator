# HiP-CT Segmentator

**HiP-CT Segmentator** is a small software tool for semi-automatic organ-background segmentation of HiP-CT scans.

## Features

- Semi-automatic segmentation of organs and background in HiP-CT datasets.
- Designed for high-resolution medical imaging data.
- Built in Python for flexibility and integration with scientific workflows.

## Installation

First, clone the repository:

```bash
git clone https://github.com/LJMedPhys/HiP-CT-segmentator.git
cd HiP-CT-segmentator
```

Install the required Python dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

### Requirements

The main dependencies are:

- `glymur==0.14.3`
- `h5py==3.11.0`
- `matplotlib==3.10.3`
- `numba==0.61.2`
- `numpy==2.3.1`
- `Pillow==11.2.1`
- `PyYAML==6.0.2`
- `tqdm==4.66.5`

## Usage

First you follow the instruction in the Segmenting_HiP.ipynb notebook. In this note book you test out the segmentation hyperparameters on a select few of slices. 

After completing the notebook and running the last cell, it exports the parameters into the config_seg file.

Then you can apply the found parameters to the whole scan an save the segmentation in an h5 file:

```bash
python run_segmentation.py --config path/to/config --input path/to/input/data --output path/to/output/file.h5 --ncpus number_of_cores
```

The segmenator accepts J2P, tiff or a single h5 file. The ncpus determines the amount of cpu cores that are used to run the segmentation.

