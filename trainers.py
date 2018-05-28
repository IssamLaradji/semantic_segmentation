import torch 
from torch.utils import data
import utils_misc as ut 
import numpy as np
from tqdm import tqdm 
import validate as val
from torch.autograd import Variable

import utils_main as mu
import time 
import tqdm

from torch.utils.data.sampler import SubsetRandomSampler
def fit(model, trainloader, opt, loss_name, 
        metric_name, iter2dis=None, verbose=1, epoch=0):
      
    loss_function = mu.LOSS_DICT[loss_name]

    n_samples = len(trainloader.dataset)
    n_batches = len(trainloader) 
    iter2val = n_batches // min(5, n_batches)

    if iter2dis is None:
      iter2dis = n_batches // min(10, n_batches)
  
    if verbose==2:
      pbar = tqdm.tqdm(total=len(trainloader), leave=False)
    elif verbose==1:
      print("Training Epoch {} .... {} batches".format(epoch, n_batches))

    # %%%%%%%%%%% 1. Train Phase %%%%%%%%%%%%"
    s_time = time.time()

    avg_meter = val.AvgMeter()
   
    for i, batch in enumerate(trainloader):
        # 1. Update
        opt.zero_grad()
        

        loss = loss_function(model, batch)
        loss.backward()

        opt.step()

        # 2. Validate
        if i % iter2val == 0:
            score = val.valBatch(model, batch, metric_name=metric_name)            
            avg_meter.update(name=metric_name, score=score)

        # 3. Details
        elapsed = ((time.time() - s_time) / 60)
        avg_meter.update(name=loss_name, score=loss.item())
        
        if verbose==2:
            pbar.set_description("{}. {} - n: {} - {}".format(epoch, 
                                 trainloader.dataset.split, 
                                 n_batches, avg_meter))
            pbar.update(1)

        elif verbose==1 and (i % iter2dis) == 0:
          print("{} - ({}/{}) - {} - {} - elapsed: {:.3f}".format(epoch,  i, n_batches, 
                trainloader.dataset.split, avg_meter, elapsed))

    if verbose==2:
        pbar.close()

    if verbose:
      ut.pprint("{}. train".format(epoch), avg_meter, 
            "n_samples: {}".format(n_samples), 
              "n_batches: {}".format(n_batches))
    
    # train: save history
    train_dict = avg_meter.get_dict()
    train_dict["epoch"] = epoch
    train_dict["n_samples"] = n_samples
    train_dict["time (min)"] = elapsed
    train_dict["iterations"] = n_batches
 
    return train_dict
    
    
def fitQuick(model, train_set, loss_name, 
        metric_name, opt=None, num_workers=1, batch_size=1, 
        verbose=1, epochs=10, n_samples=1000):

  if opt is None:
      opt = torch.optim.Adam(model.parameters(), lr=1e-3)

  ind = np.random.randint(0, len(train_set), min(n_samples, len(train_set)))
  trainloader = data.DataLoader(train_set, 
                              num_workers=num_workers,
                              batch_size=batch_size, 
                              sampler=SubsetRandomSampler(ind))
  for e in range(epochs):
    fit(model, trainloader, opt, loss_name, 
            metric_name, verbose=verbose, epoch=e)



def fitIndices(model, train_set, opt, loss_name, 
        metric_name, num_workers, batch_size, 
        verbose=1, epoch=0, ind=None):
  trainloader = data.DataLoader(train_set, 
                              num_workers=num_workers,
                              batch_size=batch_size, 
                              sampler=SubsetRandomSampler(ind))
  
  return fit(model, trainloader, opt, loss_name, 
          metric_name, verbose=verbose, epoch=epoch)
# def fitIndices(model, dataset, loss_function, indices, opt=None, epochs=10,  
#                verbose=1):
#     if opt is None:
#       opt = torch.optim.Adam(model.parameters(), lr=1e-5)

#     for epoch in range(epochs):
#       if verbose == 1:
#         pbar = tqdm(total=len(indices), leave=True)

#       lossSum = 0.
#       for i, ind in enumerate(indices):
#         batch = ut.get_batch(dataset, [ind])

#         opt.zero_grad()
#         loss = loss_function(model, batch)       
#         loss.backward()
#         opt.step()

#         lossSum += float(loss)
#         lossMean = lossSum / (i + 1)

#         if verbose == 1:
#           pbar.set_description("{} - loss: {:.3f}".format(epoch, lossMean))
#           pbar.update(1)

#         elif verbose == 2:
#           print("{} - ind:{} - loss: {:.3f}".format(epoch, ind, lossMean))

#       if verbose == 1:
#         pbar.close()
import math
def fitBatch(model, batch, loss_name=None, loss_function=None, opt=None, loss_scale="linear",
             epochs=10, verbose=2):
  if loss_function is None:
    loss_function = mu.LOSS_DICT[loss_name]

  model_name = type(model).__name__
  if verbose == 1:
    pbar = tqdm.tqdm(total=epochs, leave=False)
  if opt is None:
      opt = torch.optim.Adam(model.parameters(), lr=1e-5)

  for i in range(epochs):           
      #train_set.evaluate_count(model, batch)
      # 1. UPDATE MODEL
      opt.zero_grad()

      loss = loss_function(model, batch)       
      loss.backward()
      opt.step()
      

      loss_value = float(loss)
      if loss_scale == "log":
        loss_value = math.log(loss_value)

      if verbose == 1:
        pbar.set_description("{}: {:.3f}".format(loss_name, loss_value))
        pbar.update(1)


      elif verbose == 2:
          print("{} - {} - {}: {:.3f}".
            format(i, model_name, loss_name, loss_value))

  if verbose == 1:
    pbar.close()
  print("{} - {} - {}: {:.3f}".format(i, 
    model_name, loss_name, loss_value))

def fitBatchList(model, batchList, opt, name="", 
    verbose=True):
  
  lossSum = 0.

  if verbose:
    pbar = tqdm(total=len(batchList), leave=False)

  for i in range(len(batchList)):  
      batch = batchList[i]
      #train_set.evaluate_count(model, batch)
      # 1. UPDATE MODEL
      opt.zero_grad()
      loss = model.compute_loss(batch)                
      loss.backward()
      opt.step()

      lossSum += float(loss)
      lossMean = lossSum / (i+1)
      if verbose:
        if name != "":
          pbar.set_description("{} - loss: {:.3f}".format(name, lossMean))
        else:
          pbar.set_description("loss: {:.3f}".format(lossMean))

        pbar.update(1)
      #print("{} - loss: {:.3f}".format(i, float(loss)))

  if verbose:
    pbar.close()

    if len(batchList) > 0:
      if name != "":
        print("{} - loss: {:.3f}".format(name, lossMean))
      else:
        print("loss: {:.3f}".format(lossMean))

      

    else:
      print("{} batch is empty...".format(name))

  if len(batchList) > 0:
    return lossMean
  

def fitData(model, dataset, opt=None, epochs=10, batch_size=10):
    loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=min(batch_size, 3), shuffle=True, drop_last=True)

    n_batches = len(loader)

    for epoch in range(epochs):
      pbar = tqdm(total=n_batches, leave=False)

      lossSum = 0.
      for i, batch in enumerate(loader):
        opt.zero_grad()
        loss = model.compute_loss(batch)
        loss.backward()
        opt.step()

        lossSum += float(loss)
        lossMean = lossSum / (i + 1)

        pbar.set_description("{} - loss: {:.3f}".format(epoch, lossMean))
        pbar.update(1)

      pbar.close()

      print("{} - loss: {:.3f}".format(epoch, lossMean))

