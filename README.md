# ECE228_SP25
## Dataset
CIFAKE dataset in huggingface: https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset

## Required packages
To successfully run the code, ensure that you have the following packages installed.

```
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split, Dataset
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import ViTImageProcessor
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import seaborn as sns
import pandas as pd
import json, os
from torchdiffeq import odeint, odeint_adjoint
```
## Instructions for running the code
User should interact with the code through the **CIFAKE_base_test.ipynb** only. The third cell mount the google drive to the colab, so you should ignore it if you are running the code locally. <ins>In the fourth cell, modify the path to the path to the **models** and **utils** folder. </ins>

We have provided all trained models to reproduce the results. All models are stored in the **save_models_data** or **save_models** folders. A breif description of these models are given as follows:

- AugmentedNeuralODE_conv_data.pt: augmented NODE trained on CIFAKE. Reproduces table 1, 2, figure 2 and 3.
- CNN_base_data.pt: vanilla CNN trained on CIFAKE. Reproduces table 1, 2, figure 2 and 3.
- Densenet121_custom_data.pt: Densenet121 fine-tuned on CIFAKE. Reproduces table 1, 2, figure 2 and 3.
- Densenet201_custom_data.pt: Densenet201 fine-tuned on CIFAKE.
- ViT_pretrained_CIFAKE_data.pt: ViT finetuned on CIFAKE. Reproduces table 1, 2, figure 2 and 3.
- CNN_adversarial.pt: vanilla CNN with deep ensemble trained on CIFAKE. Reproduces table 3 and figure 4.
- Densenet121_custom_adversarial.pt: Densenet121 with deep ensemble trained on CIFAKE. Reproduces table 3 and figure 4.
- ViT_pretrained_adversarial.pt: ViT with deep ensemble trained on CIFAKE. Reproduces table 3 and figure 4.

Be sure to modify the **save_dir** variable to point to the directory that stores the model. Then, you can load the model using a subsequent cell.

The train and test loss and test accuracy in each epoch are stored as json files in the folder with the same name as the model. 

## Run the test
There are 2 cells for testing. For the 1st, 2nd, 3rd, 4th models, one should run the first cell, with the criterion argument in the test_model function being set to nn.BCELoss().

For the 5th model, one should run the second cell, with the criterion argument in the test_model function being set to nn.BCELoss().

For the 6th and 7th models, one should run the first cell, with the criterion argument in the test_model function being set to negative_log_likelihood.

For the 8th model, one should run the second cell, with the criterion argument in the test_model function being set to negative_log_likelihood.

## Visualize the results
The rest of the cells are used for plotting.
