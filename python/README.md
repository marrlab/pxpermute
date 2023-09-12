# PxPermute

## Overview

The Channel Interpretation Python Package is a tool that allows you to perform channel interpretation 
using the `PxPermute` method as well as adapted pixel-wise interpretation methods implemented via captum library.
It helps you analyze and interpret the contributions of individual channels in your data.

## Installation

1. First, make sure you have [poetry](https://python-poetry.org/) installed.

2. Set up a virtual environment and install the package dependencies using Poetry:

   ```bash
   poetry install
   ```

## Usage
For using the package, your data has to have the structure described in [scifAI documentation](https://github.com/marrlab/scifAI#data-structure).
1. Before running the interpretation method on your data, update a configuration file (config.cfg) according to your needs. 
   You can find an example of the configuration file in the repository.
   The configuration file consists of the following sections:
   - **[general]** - contains parameters like device, batch_size ad num_workers.
   - **[data]** - contains information about the data you want to interpret: path to the data file, 
   the channels to interpret with the corresponding names and other parameters to normalize your data.
   - **[interpretation]** - contains information about the interpretation method you want to use. 
   You can specify the shuffle_times for PxPermute, and some other parameters used for captum interpretation methods.
   - **[model]** - contains path to the model. 
   
2. Run the package using the following command:
   ```bash
   poetry run python main.py --method <interpretationn_method> --output-path <path_to_output_file>
   ```
   Per default, the output file will be saved in the `output` folder, and the method will be `PxPermute`.
3. The output are plots showing the contributions of individual channels in your data. 
4. Currently, the package works only with ResNet-18 model. To run interpretation methods on your own model, 
   you have to adjust the `load_model` function in `util.py` file. 
   The function should return the model.

## Interpretation Methods

Choose from a variety of interpretation methods to analyze your data:

- PxPermute
- DeepLift
- GuidedGradCam
- Saliency
- DeepLiftShap
- GradientShap
- InputXGradient
- IntegratedGradients
- GuidedBackprop
- Deconvolution
- Occlusion
- FeaturePermutation
- ShapleyValueSampling
- Lime
- KernelShap
- LRP
