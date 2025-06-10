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
User should interact with the code through the **CIFAKE_base_test.ipynb** only.
