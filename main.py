# import sys; sys.path.append("/mnt/home/issam/Research_Ground/commons")

import matplotlib
matplotlib.use('Agg')

import debug
import torch
import pandas as pd
import argparse
import numpy as np
from itertools import product
import experiments
import os
import utils_misc as ut
import test 
import train
import parser as main_parser
import utils_main as mu

def main():
  # SET SEED
  np.random.seed(1)
  torch.manual_seed(1) 
  torch.cuda.manual_seed_all(1)

  # SEE IF CUDA IS AVAILABLE
  #print("STARTING - CUDA: %s" % torch.cuda.is_available())
  assert torch.cuda.is_available()
  print("CUDA: %s" % torch.version.cuda)
  print("Pytroch: %s" % torch.__version__)

  parser = argparse.ArgumentParser()

  parser.add_argument('-e','--exp') 
  parser.add_argument('-b','--borgy', default=0, type=int)
  parser.add_argument('-br','--borgy_running', default=0, type=int)
  parser.add_argument('-m','--mode', default="summary")
  parser.add_argument('-r','--reset', default="None")
  parser.add_argument('-g','--gpu', type=int)
  parser.add_argument('-c','--configList', nargs="+",
                      default=None)
  parser.add_argument('-l','--lossList', nargs="+",
                      default=None)
  parser.add_argument('-d','--datasetList', nargs="+",
                      default=None)
  parser.add_argument('-metric','--metricList', nargs="+",
                      default=None)


  args = parser.parse_args()
  mu.set_gpu(args.gpu)
  mode = args.mode 

  exp_dict = experiments.get_experiment(args.exp)
  
  if args.configList is None:
    configList = exp_dict["configList"] 
  else:
    configList = args.configList

  if args.metricList is None:
    metricList = exp_dict["metricList"] 
  else:
    metricList = args.metricList

  if args.datasetList is None:
    datasetList = exp_dict["datasetList"] 
  else:
    datasetList = args.datasetList

  epochs = exp_dict["epochs"] 

  if args.lossList is None:
    lossList = exp_dict["lossList"] 
  else:
    lossList = args.lossList
    

  # vis.close(env="loss")
  # vis.close(env="plots")

  

  results = {}

  for config, metric, dataset, loss_name in product(configList, 
                                                    metricList, 
                                                    datasetList, 
                                                    lossList):     


      argList = mu.get_argList(mode, dataset, config, 
                               args.reset, epochs,
                               metric, loss_name)
      main_dict = mu.argList2main_dict(main_parser.get_parser(), 
                                       argList)

      key = ("%s - %s" % (config, loss_name), 
             "%s_(%s)" % (dataset, metric))


      if mode == "summary":
        results[key] = mu.summary(argList, main_dict, which="train")

      # TRAIN CASES
    
      if mode == "debug":
        debug.debug(main_dict)

      if mode == "train":
        train.main(main_dict)
         
      # TEST CASES
      if mode == "test":
        print(main_dict["dataset_name"])
        cm = test.test(main_dict)
        print(cm)

       
  print(ut.dict2frame(results))

if __name__ == "__main__":
    main()