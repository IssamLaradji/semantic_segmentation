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
import utils as ut
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

def debug(**main_dict):

  train_set, val_set = mu.load_trainval(main_dict)
  #batch=ut.get_batch(test_set, indices=[509]) 

  batch=ut.get_batch(train_set, indices=[3]) 
  model, opt, _ = mu.init_model_and_opt(main_dict)
  import ipdb; ipdb.set_trace()  # breakpoint 2167961a //
  
  tr.fitBatch(model, batch, loss_name=main_dict["loss_name"], opt=opt)
  vis.images(batch["images"], model.predict(batch, "probs"), denorm=1)
