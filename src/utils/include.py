# system
import os
import sys
import time
from datetime import datetime
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__).replace('src/utils',''))
IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# numerical libs
import math
import numpy as np
import random
import PIL
import cv2

# graph
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('WXAgg')
#matplotlib.use('Qt4Agg')
#matplotlib.use('Qt5Agg') #Qt4Agg
#print('matplotlib.get_backend : ', matplotlib.get_backend())
#print(matplotlib.__version__)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# torch libs
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import *
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.parallel.data_parallel import data_parallel


#pretrainmodels
import pretrainedmodels

# std libs
import collections
import copy
import numbers
import inspect
import shutil
from timeit import default_timer as timer
import itertools
from collections import OrderedDict
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm

# show message
from pprint import pprint, pformat

# get args from command line
import argparse

# medical image
import pydicom

# file manipulation
import csv
import pandas as pd
import pickle
import glob
import json
import zipfile
from distutils.dir_util import copy_tree
import functools

#augmentation tools
import albumentations as A
from albumentations.pytorch import ToTensor
import pretrainedmodels

#fp16
from apex import amp

#metric
from sklearn.metrics import f1_score, roc_auc_score, log_loss

# constant
PI  = np.pi
INF = np.inf
EPS = 1e-12