import sys; sys.path.append("/mnt/home/issam/Research_Ground/commons")

import matplotlib
matplotlib.use('Agg')

import sys
import argparse
import os
import utils as ut
import numpy as np
import torch

import utils_main as mu
import train, debug, test

def main(argList=None):
    # SET SEED
    np.random.seed(1)
    torch.manual_seed(1) 
    torch.cuda.manual_seed_all(1)

    # SEE IF CUDA IS AVAILABLE
    #print("STARTING - CUDA: %s" % torch.cuda.is_available())
    assert torch.cuda.is_available()
    print("CUDA: %s" % torch.version.cuda)
    
    ####################
    # GET MAIN ARGUMENTS
    ####################
    main_dict = mu.argList2main_dict(argList)
    mode = main_dict["mode"]

    if mode == "train":
        train.main(**main_dict)

    elif mode == "debug":
        debug.debug(**main_dict)

    elif mode == "test":
        test.test(**main_dict)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-mode','--mode') 

    parser.add_argument('-c','--config_name') 

    parser.add_argument('-m','--model_name')  
    parser.add_argument('-d','--dataset_name') 
    parser.add_argument('-me','--metric_name')
    parser.add_argument('-o','--opt_name') 

    parser.add_argument('-do','--dataset_options', type=ut.strList2dict, default={}) 
    parser.add_argument('-mo','--model_options', type=ut.strList2dict, default={})  
    parser.add_argument('-oo','--opt_options', type=ut.strList2dict, default={}) 

    parser.add_argument('-l','--loss_name') 
    parser.add_argument('-s','--sampler_name')
    
    parser.add_argument('-n','--epochs', type=int) 
    parser.add_argument('-e2v','--epoch2val', type=int) 
    parser.add_argument('-i2v','--iter2val', type=int) 
    
    parser.add_argument('-b','--batch_size', type=int)
    parser.add_argument('-vb','--val_batchsize', type=int)  

    parser.add_argument('-tr_transform','--trainTransformer') 
    parser.add_argument('-te_transform','--testTransformer') 
    
    parser.add_argument('-r','--reset', type=str) 
    parser.add_argument('-g','--gpu', type=int)
    parser.add_argument('-v','--verbose',  type=int) 

    parser.add_argument('-op','--options', type=dict) 

    parser.add_argument('-dl','--datasetList')

    return parser 


if __name__ == "__main__":
    main()


