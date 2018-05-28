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
from core import losses
from core import metrics
from skimage.segmentation import find_boundaries
from torch.utils.data.sampler import SubsetRandomSampler




def debug(main_dict):
  import ipdb; ipdb.set_trace()  # breakpoint 69b4b68e //
  
  metric_name = main_dict["metric_name"]
  loss_name = main_dict["loss_name"]
  batch_size = main_dict["batch_size"]
  mu.print_welcome(main_dict)

  train_set, val_set = mu.load_trainval(main_dict)
  
  model, opt, _ = mu.init_model_and_opt(main_dict)
  print("Model from scratch...")

  batch = ut.get_batch(train_set, indices=[15]) 

  probs = model.predict(batch, "probs")
  blobs = model.predict(batch, "blobs")
  tr.fitBatch(model, batch, loss_name=loss_name, opt=opt, epochs=100)
  val.valBatch(model, batch)


  import ipdb; ipdb.set_trace()  # breakpoint 5cd16f8f //
  ul.visSp_prob(model, batch)
  
  
  vis.images(batch["images"], aa, denorm=1)

  vis.visBlobs(model, batch)
  ul.vis_nei(model,batch,topk=1000, thresh=0.8,bg=True)
  ul.vis_nei(model,batch,topk=1000, bg=False)
  tr.fitQuick(model, train_set, batch_size=batch_size,loss_name=loss_name, metric_name=metric_name)
  val.validate(model, val_set, metric_name=main_dict["metric_name"], batch_size=main_dict["val_batchsize"])
  tr.fitQuick(model, train_set, batch_size=batch_size,loss_name=loss_name, metric_name=metric_name)
  tr.fitBatch(model, batch, loss_name=loss_name, opt=opt, epochs=100)
  val.valBatch(model, batch_train, metric_name=metric_name)
  tr.fitBatch(model, batch, loss_function=losses.expand_loss, opt=opt, epochs=100)
  vis.visBlobs(model, batch)
  vis.visWater(model,batch)
  val.validate(model, val_set, metric_name="MUCov")
  import ipdb; ipdb.set_trace()  # breakpoint ddad840d //
  model, opt, _ = mu.init_model_and_opt(main_dict)
  tr.fitBatch(model, batch, loss_name="water_loss_B", opt=opt, epochs=100)

  tr.fitQuick(model, train_set, loss_name=loss_name, metric_name=metric_name)
  # vis.images(batch["images"], batch["labels"], denorm=1)
  # mu.init.LOSS_DICT["water_loss"](model, batch)
  import ipdb; ipdb.set_trace()  # breakpoint f304b83a //
  vis.images(batch["images"], model.predict(batch, "labels"), denorm=1)
  val.valBatch(model, batch, metric_name=main_dict["metric_name"])
  vis.visBlobs(model, batch)
  import ipdb; ipdb.set_trace()  # breakpoint 074c3921 //

  tr.fitBatch(model, batch, loss_name=main_dict["loss_name"], opt=opt, epochs=100)
  for e in range(10):
    if e == 0:
      scoreList = []
    scoreList += [tr.fitIndices(model, train_set, loss_name=main_dict["loss_name"], batch_size=batch_size,
      metric_name=metric_name, opt=opt, epoch=e, num_workers=1, 
      ind=np.random.randint(0, len(train_set), 32))]
  tr.fitData(model, train_set, opt=opt, epochs=10)
  ut.reload(sp);water=sp.watersplit(model, batch).astype(int);vis.images(batch["images"], water, denorm=1)
  vis.visBlobs(model, batch)
  vis.images(batch["images"], ul.split_crf(model, batch),denorm=1)
  losses.dense_crf(model, batch, alpha=61, beta=31, gamma=1)
  
  vis.visBlobs(model, batch)

  model.blob_mode = "superpixels"
  #----------------------

  # Vis Blobs
  vis.visBlobs(model, batch)
  vis.images(batch["images"],model.predict(batch, "labels"), denorm=1)

  # Vis Blobs
  #vis.visBlobs(model, batch)
  vis.images(batch["images"], sp.watersplit_test(model, batch).astype(int), denorm=1)

  #=sp.watersplit(model, batch).astype(int);

  # Vis CRF
  vis.images(batch["images"], ul.dense_crf(model, batch, alpha=5,gamma=5,beta=5,smooth=False), denorm=1)
  vis.images(batch["images"], ul.dense_crf(model, batch), denorm=1)
  # Eval
  val.valBatch(model, batch, metric_name=main_dict["metric_name"])

  import ipdb; ipdb.set_trace()  # breakpoint e9cd4eb0 //
  model = mu.load_best_model(main_dict)

  val.valBatch(model, batch, metric_name=main_dict["metric_name"])
  tr.fitBatch(model, batch, loss_name=main_dict["loss_name"], opt=opt)
  vis.visBlobs(model, batch)
  import ipdb; ipdb.set_trace()  # breakpoint 2167961a //
  batch=ut.get_batch(train_set, indices=[5]) 
  tr.fitBatch(model, batch, loss_name=main_dict["loss_name"], opt=opt)
  vis.images(batch["images"], model.predict(batch, "probs"), denorm=1)

  vis.visBlobs(model, batch)
  val.validate(model, val_set, metric_name=main_dict["metric_name"])
  val.validate(model, val_set, metric_name="SBD")



def plot_density():
  import pylab as plt
  import numpy as np

  # Sample data
  side = np.linspace(-2,2,15)
  X,Y = np.meshgrid(side,side)
  Z = np.exp(-((X-1)**2+Y**2))
  fig = plt.figure()
  # Plot the density map using nearest-neighbor interpolation
  plt.pcolormesh(X,Y,Z)
  vis.visplot(fig)