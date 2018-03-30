import sys; sys.path.append("/mnt/home/issam/Research_Ground/commons")

import torch
import main
import pandas as pd
import argparse
import utils as ut
import numpy as np
from itertools import product
import utils_main as mu
import experiments
import os
import trainers as tr
import validate as val
import utils_borgy as bu
import utils_script as su
import test 
from collections import defaultdict
import train


# def dict2frame(myDict):
#   # myDict = {('a','b'):10, ('a','c'):20}
#   # data = list(map(list, zip(*myDict.keys())) + [myDict.values()])
#   # df = pd.DataFrame(zip(*data)).set_index([0, 1])[2].unstack()
#   # return df.combine_first(df.T)

#   s = pd.Series(myDict, index=pd.MultiIndex.from_tuples(myDict))
#   df = s.unstack()
#   lhs, rhs = df.align(df.T)
#   res = lhs.add(rhs, fill_value=0)

#   return res 
import debug
def script():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e','--exp') 
    parser.add_argument('-m','--mode', default="summary")
    parser.add_argument('-r','--reset', default="None")
    parser.add_argument('-g','--gpu', type=int)

    args = parser.parse_args()
    mode = args.mode 

    exp_dict = experiments.get_experiment(args.exp)

    configList = exp_dict["configList"] 
    metricList = exp_dict["metricList"] 
    datasetList = exp_dict["datasetList"] 
    epochs = exp_dict["epochs"] 

    # vis.close(env="loss")
    # vis.close(env="plots")

    mu.set_gpu(args.gpu)

    results = {}
    
    for config in configList:
      for metric, dataset in product(metricList, datasetList):     

        argList = ("-mode %s -d %s \
                -c %s -r %s -n %d -me %s" % 
                (mode, dataset, config, args.reset, epochs, metric))


        # 1. EXTRACT INFO
        main_dict = mu.argList2main_dict(argList)

        key = (config, "%s_(%s)" % (dataset, metric))

        if mode == "debug":
          debug.debug(**main_dict)
          import ipdb; ipdb.set_trace()  # breakpoint 9a857e2d //
        if mode == "figures":
          figures.figures(main_dict)

        # TRAIN CASES
        if mode in ["train", "borgy", "summary", "status", "kill"]:
          if mode == "train":
            main.main(argList)
            import ipdb; ipdb.set_trace()  # breakpoint a5d091b9 //


          if mode == "summary":
            results[key] = su.summary(argList, main_dict, which="train")

          if mode == "status":
            train_command = bu.get_command(argList, mode="train")
            job_id, job_state = bu.get_job_id(train_command)
            results[key] = "%s - %s" % (job_state, job_id)

          if mode == "borgy":
            train_command = bu.get_command(argList, mode="train")
            print(train_command)

            job_id, job_state = bu.get_job_id(train_command)

            if bu.job_is_running(job_state):
              results[key] =  "%s - %s" % (job_state, job_id)

            else:
              prompt = input("Do you want to borgy submit the command:"
                     " \n'%s' ? \n(y/n)\n" % train_command) 
              if prompt == "y":            
                if not su.is_exist_train(main_dict):              
                  print(bu.borgy_submit(train_command, force=True))

          if mode == "kill":   
            train_command = bu.get_command(argList, mode="train")
            job_id, job_state = bu.get_job_id(train_command)
            bu.borgy_kill(job_id, force=True)   

        # TEST CASES
        if mode in ["test", "borgy_test", "summary_test", 
                    "status_test"]:
          if mode == "test":
            results[key] = test.test_count(**main_dict)

          if mode == "status_test":
            test_command = bu.get_command(argList, mode="test_count")
            job_id, job_state = bu.get_job_id(test_command)
            results[key] = "%s - %s" % (job_state, job_id)

          if mode == "summary_test":
            results[key] = su.summary(argList, main_dict, which="test_count")

          if mode == "borgy_test":   
            test_command = bu.get_command(argList, mode="test_count")
            
            if not test.is_exist(main_dict):
              print(bu.borgy_submit(test_command, force=True))

            job_id, job_state = bu.get_job_id(test_command)

            if bu.job_is_running(job_state):
              results[key] = job_state

            else:
              results[key] = su.summary(argList, main_dict, which="test_count")



        # FSCORE CASES
        if mode in ["borgy_fscore", "fscore", "summary_fscore"]:
          if main_dict["model_name"] == "Glance":
            continue
          if mode == "fscore":
            results[key] = test.fscore(**main_dict)

          if mode == "summary_fscore":
            try:
              results[key] = su.summary(argList, main_dict, which="fscore")
            except:
              results[key] = bu.get_job_id(bu.get_command(argList, mode="fscore"))

          if mode == "borgy_fscore":
            test_command = bu.get_command(argList, mode="fscore")
            
            if not test.is_fscore_exist(main_dict):
              print(bu.borgy_submit(test_command, force=True))

            job_id, job_state = bu.get_job_id(test_command)

            if bu.job_is_running(job_state):
              results[key] = job_state

            else:
              results[key] =  su.summary(argList, main_dict, which="fscore")


         
    print(su.dict2frame(results))

if __name__ == "__main__":
    script()