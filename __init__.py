import torch.optim as optim

import utils as ut

from importlib import import_module

import datasets
import models

from base import losses
from base import metrics 
from base import samplers



LOSS_DICT = ut.get_functions(losses)
METRIC_DICT = ut.get_functions(metrics)
SAMPLER_DICT = ut.get_functions(samplers)


OPT_DICT = {"adam":optim.Adam, 
            "adamFast":lambda params, lr, 
            weight_decay:optim.Adam(params, lr=lr, betas=(0.9999,0.9999999), weight_decay=weight_decay),

            "sgd":lambda params, lr, 
            weight_decay:optim.SGD(params, lr=lr, weight_decay=weight_decay,
                                                    momentum=0.9)}


###############
import inspect
def get(folder):
  mod_dict = {}
  modList = [import_module("%s.%s" % (folder, class_name)) 
              for class_name in eval("%s.__all__"%folder)]

  for module in modList:
    funcs = ut.get_functions(module)
    for name in funcs:
      val = funcs[name]

      if not inspect.isclass(val):
        continue


      if (name in mod_dict and 
         folder in str(val.__module__)):
         if name != "Pascal2012":
            raise ValueError("repeated %s" % name)
         print("Repeated:", name)
      mod_dict[name] = val

  return mod_dict
# DATASETS

DATASET_DICT = get("datasets")
###############
# MODELS

MODEL_DICT = get("models")

