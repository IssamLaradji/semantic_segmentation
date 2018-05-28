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



import datasets, models

import datetime as dt
import time
import pprint
import validate as val
import utils_misc as ut
import pprint

PATH_DATASETS = ut.load_json("meta.json")["path_datasets"]
PATH_SAVES = ut.load_json("meta.json")["path_saves"]


try:
  from core import losses
  LOSS_DICT = ut.get_functions(losses)
except:
  LOSS_DICT = {}

try:
  from core import metrics
  METRIC_DICT = ut.get_functions(metrics)
except:
  METRIC_DICT = []

try:
  from core import samplers
  SAMPLER_DICT = ut.get_functions(samplers)
except:
  SAMPLER_DICT = {}



# DATASETS
DATASET_DICT = ut.get_modules("datasets")
MODEL_DICT = ut.get_modules("models")

# Optimizers 
OPT_DICT = {"adam":optim.Adam, 
            "adamFast":lambda params, lr, 
            weight_decay:optim.Adam(params, lr=lr, betas=(0.9999,0.9999999), weight_decay=weight_decay),

            "sgd":lambda params, lr, 
            weight_decay:optim.SGD(params, lr=lr, weight_decay=weight_decay,
                                                    momentum=0.9)}




def summary(argList, main_dict, which):
  history = load_history(main_dict)
  if history is None:
    return "None"
  metric_name = main_dict["metric_name"]
  
  if "epoch" not in history["best_model"]:
    return "Not Yet"


  best_epoch = history["best_model"]["epoch"]
  epoch = history["epoch"]

  if which == "train":
    try:
      loss = history["train"][-1][main_dict["loss_name"]]
    except:
      loss = "Not Found"
    #loss = 1
    best_score =  history["best_model"][metric_name]
    score = ("loss: {:.3} | ({}/{}) {:.3f}".format 
             ( loss, best_epoch, epoch, best_score))

  if which == "test_count":
    fname = main_dict["path_save"] + "/test_count.pkl"

    records = ut.load_pkl(fname)
    if best_epoch != records["best_epoch"]:
      state = "* "
    else:
      state = " "
    score = "({}/{}){}{}".format(best_epoch, epoch, state,
                              records[metric_name])

  if which == "fscore":
    fname = main_dict["path_save"] + "/test_fscore.pkl"

    records = ut.load_pkl(fname)
    if best_epoch != records["best_epoch"]:
      state = "* "
    else:
      state = " "
    score = "({}/{}){}{}".format(best_epoch, epoch, state,
                              records["fscore"])

  return score
  
def get_metric_func(main_dict=None, metric_name=None):
  if metric_name is not None:
    return METRIC_DICT[metric_name]
  return METRIC_DICT[main_dict["metric_name"]]



def get_argList(mode, dataset, config, reset, epochs,
                metric, loss_name):
        argList = ("-mode %s -d %s \
              -c %s -r %s -n %d -me %s -l %s" % 
              (mode, dataset, config, reset,
               epochs, metric, loss_name))

        return argList

def val_test(main_dict, metric_name=None):
  test_set = load_test(main_dict)

  try:
    model = load_best_model(main_dict)
  except:
    model, _, _ = init_model_and_opt(main_dict)

  if metric_name is None:
    metric_name=main_dict["metric_name"]

  score = val.validate(model, test_set,
                       metric_name=metric_name)
  return score 


def prettyprint(main_dict):
      pprint.PrettyPrinter(depth=6).pprint(
            {k:main_dict[k] for k in main_dict 
                            if main_dict[k] is not None})
def print_welcome(main_dict):
    pprint.PrettyPrinter(depth=6).pprint(
            {k:main_dict[k] for k in main_dict 
                            if main_dict[k] is not None})

    ut.print_header("EXP: %s,  Reset: %s" % 
                   (main_dict["exp_name"], 
                    main_dict["reset"]))

def set_gpu(gpu_id):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]="%d" % gpu_id

def argList2main_dict(parser, argList):
    # CHECK IF ARG LIST IS PROVIDED
    if argList is None:
        main_args = parser.parse_args()
    else:
        main_args = ut.parse_command(argList, parser)

    main_dict = vars(main_args)

    override_dict = {key:main_dict[key] for key in main_dict 
                     if (main_dict[key] is not None and
                         main_dict[key] != {} and
                         main_dict[key] != [])}

    # GET GPU
    set_gpu(main_args.gpu)

    command = ut.load_json("configs.json")[ main_dict["config_name"]]

    config_dict = vars(ut.parse_command(command, parser))
    for key in config_dict:
        main_dict[key] = config_dict[key]

    for key in override_dict:
        main_dict[key] = override_dict[key]

    main_dict["path_datasets"] = PATH_DATASETS

    ############## GET EXP NAME
    main_dict["exp_name"] = get_exp_name(main_dict)


    # SAVE
    main_dict["path_save"] = ("%s/%s/" % 
                              (PATH_SAVES, 
                               main_dict["exp_name"]))
    main_dict["path_summary"] = main_dict["path_save"].replace("Saves", "Summaries")
    return main_dict

def get_exp_name(main_dict):
  config_name = main_dict["config_name"]
  metric_name = main_dict["metric_name"]
  dataset_name = main_dict["dataset_name"]
  loss_name = main_dict["loss_name"]

  exp_name = ("%s_dataset:%s_metric:%s_loss:%s" % 
             (config_name, dataset_name, 
              metric_name, loss_name))

  return exp_name

#### DATASET
def get_trainloader(main_dict):
  train_set = load_trainval(main_dict, train_only=True)
  dataloader = get_dataloader(train_set, 
                              batch_size=main_dict["batch_size"], 
                              sampler_name=main_dict["sampler_name"])
  return dataloader

def get_testloader(main_dict):
  test_set = load_test(main_dict)
  
  dataloader = get_dataloader(test_set, 
                              batch_size=main_dict["val_batchsize"], 
                              sampler_name=None)
  return dataloader

def load_test_dict(main_dict):
  return ut.load_pkl(main_dict["path_save"] + "/test.pkl")

def save_test_dict(main_dict, test_dict):
  return ut.save_pkl(main_dict["path_save"] + "/test.pkl", test_dict)

def load_history(main_dict):
  if not os.path.exists(main_dict["path_save"] + "/history.pkl"):
    return None
  return ut.load_pkl(main_dict["path_save"] + "/history.pkl")

def get_dataloader(dataset, batch_size, sampler_name):
  if sampler_name is None:
    trainloader = data.DataLoader(dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=min(batch_size,2), 
                                drop_last=False)
  else:
    sampler = SAMPLER_DICT[sampler_name](dataset)
    trainloader = data.DataLoader(dataset, batch_size=batch_size, 
                                sampler=sampler, 
                                num_workers=min(batch_size,2), 
                                drop_last=False)

  return trainloader
from torch.utils.data.sampler import SubsetRandomSampler
def subsetloader(dataset, batch_size, ind, num_workers=1):
  sampler = SubsetRandomSampler(ind)
  loader = data.DataLoader(dataset, batch_size=batch_size, 
                              sampler=sampler, 
                              num_workers=min(batch_size,2), 
                              drop_last=False)
  return loader

def load_trainval(main_dict, train_only=False):  
  
  path_datasets = main_dict["path_datasets"]
  dataset_name = main_dict["dataset_name"]
  trainTransformer = main_dict["trainTransformer"]
  testTransformer = main_dict["testTransformer"]
  dataset_options = main_dict["dataset_options"]
  
  train_set = DATASET_DICT[dataset_name](root=path_datasets, 
                                         split="train", 
                                         transform_name=trainTransformer,
                                         **dataset_options)
  if train_only:
    return train_set

  val_set = DATASET_DICT[dataset_name](root=path_datasets, 
                                         split="val", 
                                         transform_name=testTransformer,
                                         **dataset_options)

  stats = [{"dataset":dataset_name, 
            "n_train": len(train_set), 
            "n_val":len(val_set)}]
            
  print(pd.DataFrame(stats))

  return train_set, val_set

def load_test(main_dict):
  path_datasets = main_dict["path_datasets"]
  dataset_name = main_dict["dataset_name"]
  testTransformer = main_dict["testTransformer"]
  dataset_options = main_dict["dataset_options"]

  test_set = DATASET_DICT[dataset_name](root=path_datasets,  
                                             split="test", 
                                             transform_name=testTransformer,
                                             **dataset_options)

  return test_set


#### MODEL INIT
def create_model(main_dict, train_set=None):
  # LOAD MODELS
  model_name = main_dict["model_name"]
  model_options = main_dict["model_options"]


  if train_set is None:
    train_set = load_trainval(main_dict, train_only=True)

  model = MODEL_DICT[model_name](train_set=train_set, 
                                      **model_options).cuda()
  return model

def create_model_and_opt(main_dict, train_set=None):
  # LOAD MODELS
  model = create_model(main_dict, train_set=train_set)

  opt_name = main_dict["opt_name"]
  opt_options = main_dict["opt_options"]

  opt = OPT_DICT[opt_name](filter(lambda p: p.requires_grad, model.parameters()), 
                                **opt_options)
  return model, opt 

def create_opt(model, main_dict, train_set=None):
  # LOAD MODELS
  opt_name = main_dict["opt_name"]
  opt_options = main_dict["opt_options"]

  opt = OPT_DICT[opt_name](filter(lambda p: p.requires_grad, model.parameters()), 
                                **opt_options)
  return opt 

def init_model_and_opt(main_dict, train_set=None):
  # SET TIME
  start_time = dt.datetime.now(dt.timezone(dt.timedelta(hours=-8.0)))
  start_time = start_time.strftime("%I:%M%p %a, %d-%b-%y")
  
  exp_name = main_dict["exp_name"]
  metric_name = main_dict["metric_name"]
  path_save = main_dict["path_save"]

  # LOAD HISTORY
  history = {"start_time": start_time,
             "exp_name":exp_name,
             "epoch":0,
             "metric_name":metric_name,
             "main_dict": main_dict,
             "path_history":path_save + "/history.pkl",
             "train": [],
             "val": [],
             "path_train_opt":path_save + "/State_Dicts/opt.pth",
             "path_train_model":path_save + "/State_Dicts/model.pth",
             "path_best_model": path_save + "/State_Dicts/best_model.pth",
             "best_model":{}}

  model, opt = create_model_and_opt(main_dict, train_set)

  return model, opt, history


# LOADING AND SAVING MODELS
def load_latest_model(main_dict, train_set=None):
  model = create_model(main_dict, 
                      train_set=train_set)

  history = ut.load_pkl(main_dict["path_save"] + "/history.pkl")
  
  model.load_state_dict(torch.load(history["path_train_model"]))
  
  return model 

def load_latest_model_and_opt(main_dict, train_set=None):
  model, opt = create_model_and_opt(main_dict, 
                                    train_set=train_set)

  history = ut.load_pkl(main_dict["path_save"] + "/history.pkl")
  
  model.load_state_dict(torch.load(history["path_train_model"]))
  opt.load_state_dict(torch.load(history["path_train_opt"]))

  return model, opt, history

def save_latest_model_and_opt(model, opt, history):
  

  pbar = tqdm.tqdm(desc="Saving Model...Don't Exit...       ", leave=False)
  
  ut.create_dirs(history["path_train_model"])
  torch.save(model.state_dict(), history["path_train_model"])
  torch.save(opt.state_dict(), history["path_train_opt"])

  ut.save_pkl(history["path_history"], history)
  
  pbar.close()


#######################################
def load_best_model(main_dict, train_set=None):
  model = create_model(main_dict, 
                       train_set=train_set)
  print("loaded best model...")
  history = ut.load_pkl(main_dict["path_save"] + "/history.pkl")
  model.load_state_dict(torch.load(history["path_best_model"]))

  return model

def save_best_model(model, history):
  metric_name = history["metric_name"]

  pbar = tqdm.tqdm(desc="Saving Model...Don't Exit...       ", leave=False)
  
  ut.create_dirs(history["path_best_model"])
  torch.save(model.state_dict(), history["path_best_model"])

  pbar.close()

  print("New best model...%s: %.3f" % (metric_name, 
                                       history["best_model"][metric_name]))


def save_model(path, model):
  pbar = tqdm.tqdm(desc="Saving Model...Don't Exit...       ", leave=False)
  ut.create_dirs(path)
  torch.save(model.state_dict(), path)
  pbar.close()



### SUMMARY
def get_summary(main_dict):
  if os.path.exists(main_dict["path_save"] + "/history.pkl"):
    history = ut.load_pkl(main_dict["path_save"] + "/history.pkl")

    loss_name = main_dict["loss_name"]
    metric_name = main_dict["metric_name"] 
    dataset_name = main_dict["dataset_name"] 
    config_name = main_dict["config_name"] 
    
    summary = {}
    summary["config"] = config_name
    summary["dataset"] = dataset_name
    summary["metric_name"] = metric_name

    # train
    try:
      summary["_train_%s"% metric_name] = history["train"][-1][metric_name]
      summary["train_epoch"] = history["train"][-1]["epoch"]
      summary[loss_name] = "%.3f" % history["train"][-1][loss_name]
    except:
      pass

    # val
    try:
      epoch = history["val"][-1]["epoch"]
      score = history["val"][-1][metric_name]
      summary["val"] = ("%d-%.3f" %
                        (epoch, score))

      epoch = history["best_model"]["epoch"]
      score = history["best_model"][metric_name]
      summary["val_best"] = ("%d-%.3f" %
                             (epoch, score))
    except:
      pass

    return summary

  else:
    return {}

