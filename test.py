import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import sys
import os
import os.path as osp
import datetime
import random
import timeit, tqdm
import validate as val
import trainers as tr
import utils_main as mu
import pandas as pd 
from pydoc import locate
start = timeit.default_timer()
import datetime as dt
import time
from core import losses
from skimage.segmentation import find_boundaries
from sklearn.metrics import confusion_matrix



def test(main_dict, metric_name=None, save=False):
    with torch.no_grad():
        print("%s - %s - %s" % (main_dict["dataset_name"],
                                main_dict["config_name"], 
                                main_dict["loss_name"]))
        if metric_name is None:
            metric_name = main_dict["metric_name"]
   
        score = mu.val_test(main_dict, 
                            metric_name=metric_name)

            
        return score
