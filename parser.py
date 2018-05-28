import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-mode','--mode') 

    parser.add_argument('-c','--config_name') 

    parser.add_argument('-m','--model_name')  
    parser.add_argument('-d','--dataset_name') 
    parser.add_argument('-me','--metric_name')
    parser.add_argument('-o','--opt_name') 

    parser.add_argument('-do','--dataset_options', type=strList2dict, default={}) 
    parser.add_argument('-mo','--model_options', type=strList2dict, default={})  
    parser.add_argument('-oo','--opt_options', type=strList2dict, default={}) 

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
    parser.add_argument('-v','--verbose',  type=int, default=1) 

    parser.add_argument('-op','--options', type=dict) 

    parser.add_argument('-dl','--datasetList')

    return parser 



def strList2dict(string):
    strList = string.split("|")

    doDict = {}

    for l in strList:
        key = l.split(":")[0]
        val = l.split(":")[1]

        if val.isdigit():
            val = int(val)
        else:
            try:
                val = float(val)
            except ValueError:
                val = val

        doDict[key] = val

    return doDict
