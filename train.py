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
import pandas as pd 
from pydoc import locate
start = timeit.default_timer()
import datetime as dt
import time
import utils_main as mu
import utils_misc as ut
import trainers as tr
import validate as val


def main(main_dict):
  mu.print_welcome(main_dict)

  # SET SEED
  np.random.seed(1)
  torch.manual_seed(1) 
  torch.cuda.manual_seed_all(1)

  # EXTRACT VARIABLES
  reset =  main_dict["reset"]
  epochs =  main_dict["epochs"]
  batch_size = main_dict["batch_size"]
  sampler_name = main_dict["sampler_name"]
  verbose = main_dict["verbose"]
  loss_name = main_dict["loss_name"]
  metric_name = main_dict["metric_name"]
  epoch2val = main_dict["epoch2val"]
  val_batchsize = main_dict["val_batchsize"]

  # Dataset  
  train_set, val_set = mu.load_trainval(main_dict)
  train_set[0]
  # Model  
  if reset == "reset":
    model, opt, history = mu.init_model_and_opt(main_dict, 
                                                train_set) 
    print("TRAINING FROM SCRATCH EPOCH: %d/%d" % (history["epoch"],
                                                  epochs))
  else:
    model, opt, history = mu.load_latest_model_and_opt(main_dict, 
                                                       train_set) 
    print("RESUMING EPOCH %d/%d" % (history["epoch"], epochs)  ) 
  
  # Get Dataloader
  trainloader = mu.get_dataloader(dataset=train_set, 
                                  batch_size=batch_size, 
                                  sampler_name=sampler_name)

  # SAVE HISTORY
  history["epoch_size"] = len(trainloader)
  ut.save_pkl(history["path_history"], history)

  # START TRAINING
  start_epoch = history["epoch"]
  for epoch in range(start_epoch + 1, epochs):

    # %%%%%%%%%%% 1. TRAIN PHASE %%%%%%%%%%%%"    
    train_dict = tr.fit(model, trainloader, opt, loss_name, 
                        metric_name, verbose=verbose, 
                        epoch=epoch)

    # Update history
    history["epoch"] = epoch 
    history["train"] += [train_dict]

    # Save model, opt and history
    mu.save_latest_model_and_opt(model, opt, history)

    # %%%%%%%%%%% 2. VALIDATION PHASE %%%%%%%%%%%%"
    if (epoch % epoch2val == 0 or 
        epoch == 1):   
      val_dict = val.validate(dataset=val_set, 
                              model=model, 
                              verbose=verbose, 
                              metric_name=metric_name, 
                              batch_size=val_batchsize,
                              epoch=epoch)

      # Update history
      history["val"] += [val_dict]

      # Higher is better
      if (history["best_model"] == {} or 
          history["best_model"][metric_name] >= val_dict[metric_name]):

        history["best_model"] = val_dict
        mu.save_best_model(model, history)

      ut.save_pkl(history["path_history"], history)



