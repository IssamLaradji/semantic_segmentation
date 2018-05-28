import utils_misc as ut 
import numpy as np 

from torch.utils import data
from tqdm import tqdm
try:
  import utils_main as mu 
except:
  pass

from sklearn.metrics import confusion_matrix


import torch
def valBatch(model, batch, metric_name=None, metric_class=None):    
    model.eval()
    with torch.no_grad():
      assert not (metric_name is None and metric_class is None)
      if metric_class is None:
        metric_class = mu.METRIC_DICT[metric_name]

      metricObject = metric_class()
      score_dict = metricObject.scoreBatch(model, batch)
   
      return score_dict["score"]


def valLoss(model, batch, loss_name):    
    model.eval()
    with torch.no_grad():
      return mu.LOSS_DICT[loss_name](model, batch).item()

def validate(model, dataset,  
             metric_name,
             batch_size=1, epoch=0, 
             verbose=1,
             num_workers=1,
             sampler_name=None,
             sampler=None):
    batch_size = min(batch_size, len(dataset))

    if sampler_name is not None:
      sampler = mu.SAMPLER_DICT[sampler_name](dataset)

    if sampler is None:
      loader = data.DataLoader(dataset, 
                               batch_size=batch_size, 
                               num_workers=num_workers, 
                               drop_last=False)
    else:
      loader = data.DataLoader(dataset, batch_size=batch_size, 
                               num_workers=num_workers, 
                               drop_last=False,
                               sampler=sampler)

    return val(model, loader, metric_name, epoch=epoch, 
             verbose=verbose)

def val(model, loader,  metric_name, epoch=0, 
              verbose=1):

  model.eval()

  metric_class = mu.METRIC_DICT[metric_name]
  split_name = loader.dataset.split
  
  n_batches = len(loader)

  if verbose==2:
    pbar = tqdm(desc="Validating %s set (%d samples)" % 
                (split_name, n_batches), total=n_batches, leave=False)
  elif verbose==1:
    print("Validating... %d" % len(loader.dataset))

  metricObject = metric_class()
  
  iter2dis = n_batches // min(10, n_batches)

  for i, batch in enumerate(loader):
    metricObject.update_running_average(model, batch)

    #######
    progress = ("%d - %d/%d - Validating %s set - %s: %.3f" % 
               (epoch, i, n_batches, split_name, metric_name, 
                metricObject.get_running_average()))

    if verbose==2:
      pbar.set_description(progress)
      pbar.update(1)

    elif verbose==1 and i % iter2dis == 0:
      print(progress)

  if verbose==2:
    pbar.close()
  
  score = metricObject.get_running_average()

  score_dict = {}
  score_dict[metric_name] = score
  score_dict["n_samples"] = len(loader.dataset)
  score_dict["epoch"] = epoch


  # Print to screen
  if verbose:
    ut.pprint("%d - %s" % (epoch, split_name), dict2str(score_dict))
  
  score_dict["split_name"] = split_name
  return score_dict











def get_preds(model, dataset,  
             batch_size=1, epoch=0, 
             verbose=1,
             num_workers=1,
             sampler_name=None):
            
    model.eval()

    split_name = dataset.split
    batch_size = min(batch_size, len(dataset))
    
    if sampler_name is None:
      loader = data.DataLoader(dataset, batch_size=batch_size, 
                               num_workers=num_workers, 
                               drop_last=False)
    else:
      sampler = mu.SAMPLER_DICT[sampler_name](dataset)
      loader = data.DataLoader(dataset, batch_size=batch_size, 
                               num_workers=num_workers, 
                               drop_last=False,
                               sampler=sampler)

    n_batches = len(loader)

    if verbose==1:
      pbar = tqdm(desc="Validating %s set (%d samples)" % 
                  (split_name, n_batches), total=n_batches, leave=False)
    else:
      print("Validating... %d" % len(dataset))

    
    iter2dis = n_batches // min(10, n_batches)
    preds = np.ones(len(dataset))*-1
    counts = np.ones(len(dataset))*-1

    for i, batch in enumerate(loader):
      preds[i*batch_size:(i+1)*batch_size] = ut.t2n(model.predict(batch, "counts")).ravel()
      counts[i*batch_size:(i+1)*batch_size] = ut.t2n(batch["counts"]).ravel()

      #######
      progress = ("%d - %d/%d - Validating %s set" % 
                 (epoch, i, n_batches, split_name))

      if verbose==1:
        pbar.set_description(progress)
        pbar.update(1)

      elif i % iter2dis == 0:
        print(progress)

    if verbose==1:
      pbar.close()
    

    score_dict = {}
    score_dict["preds"] = preds
    score_dict["counts"] = counts
    score_dict["n_samples"] = len(dataset)
    score_dict["epoch"] = epoch


    # Print to screen
    ut.pprint("%d - %s" % (epoch, split_name), dict2str(score_dict))
    
    score_dict["split_name"] = split_name
    return score_dict
from torch.utils.data.sampler import SubsetRandomSampler

def save_images(main_dict, n_samples=10, common_path=None,
             batch_size=1,
             verbose=1,
             num_workers=1):
            
    print("%s_%s_%s_%s" % (main_dict["config_name"], main_dict["loss_name"] , 
                           main_dict["loss_name"], main_dict["loss_name"]))
    model = mu.load_best_model(main_dict)
    model.eval()
    if common_path is None:
      path = main_dict['path_summary']
    else:
      path = common_path
    assert "/Summaries/" in path
    try:
      ut.remove_dir(path)
    except:
      pass
    _, val_set = mu.load_trainval(main_dict)

    split_name = val_set.split
    batch_size = min(batch_size, len(val_set))
    np.random.seed(1)
    #sampler = SubsetRandomSampler(indices=np.random.choice(len(val_set),min(len(val_set), n_samples), replace=False))
    loader = data.DataLoader(val_set, 
                             batch_size=batch_size, 
                             num_workers=num_workers, 
                             drop_last=False)


    n_batches = len(loader)


    print("Saving... %d" % len(val_set))

    n_saves = 0
    for i, batch in enumerate(loader):
      counts = int(batch["counts"].item())
      if counts == 0:
        continue

      if n_saves == n_samples:
        break
      blobs = model.predict(batch, "blobs").squeeze()
      image = vis.get_image(batch["images"], blobs, denorm=1)
      points = vis.get_image(batch["images"],  batch["points"], enlarge=1, denorm=1)
      index = int(batch["index"].item())

    
      preds = int((np.unique(blobs)!=0).sum())

      if counts == preds:
        #print("Good")
        continue


      prefix = "%d_%s_%s_%d_%d" % (
                 index, main_dict["config_name"], main_dict["loss_name"],  preds, counts)
      ut.imsave(path + "/%s_blobs.png" % prefix, image)
      ut.imsave(path + "/%s_points.png" % prefix, points)
      #ut.imsave(path + "/%s_%s_points.png" % (index, counts))
      #######
      progress = ("%d/%d - %s" % 
                 (n_saves, n_samples, prefix))
      n_saves += 1
      print(progress)

    #loader.close()
def save_preds(main_dict, path=None,
             batch_size=1,
             verbose=1,
             num_workers=1):
            
    print("%s_%s_%s_%s" % (main_dict["config_name"], main_dict["loss_name"] , 
                           main_dict["loss_name"], main_dict["loss_name"]))
    model = mu.load_best_model(main_dict)
    model.eval()
    if common_path is None:
      path = main_dict['path_summary']
    else:
      path = common_path
    assert "/Summaries/" in path
    try:
      ut.remove_dir(path)
    except:
      pass
    _, val_set = mu.load_trainval(main_dict)

    split_name = val_set.split
    batch_size = min(batch_size, len(val_set))
    np.random.seed(1)
    sampler = SubsetRandomSampler(indices=np.random.choice(len(val_set),min(len(val_set), n_samples), replace=False))
    loader = data.DataLoader(val_set, sampler=sampler, 
                             batch_size=batch_size, 
                             num_workers=num_workers, 
                             drop_last=False)


    n_batches = len(loader)


    print("Saving... %d" % len(val_set))


    for i, batch in enumerate(loader):
      blobs = model.predict(batch, "blobs").squeeze()
      image = vis.get_image(batch["images"], blobs, denorm=1)
      points = vis.get_image(batch["images"],  batch["points"], enlarge=1, denorm=1)
      index = batch["index"].item()
      import ipdb; ipdb.set_trace()  # breakpoint 3103d145 //
      
      counts = int(batch["counts"].item())
      preds = int((np.unique(blobs)!=0).sum())

      if counts == preds:
        #print("Good")
        continue

      prefix = "%s_%s_%d_%d" % (main_dict["config_name"], main_dict["loss_name"] , index, preds, counts)
      ut.imsave(path + "/%s_blobs.png" % prefix, image)
      ut.imsave(path + "/%s_points.png" % prefix, points)
      #ut.imsave(path + "/%s_%s_points.png" % (index, counts))
      #######
      progress = ("%d/%d - Saving %s set" % 
                 (i, n_batches, split_name))

      print(progress)
def validate_stats(model, dataset, verbose=1, metric_class=None, predictFunc=None):
    model.eval()
    
    loader = data.DataLoader(dataset, batch_size=1, 
                             num_workers=1, drop_last=False)

    n_batches = len(loader)

    if verbose==1:
      pbar = tqdm(desc="Validating Test set (%d samples)" % 
                  (n_batches), total=n_batches, leave=False)

    metricObject = metric_class()
    metric_name = metric_class.__name__

    Corrects = []
    Wrongs = []
    scoreList = []
    for i, batch in enumerate(loader):
      
      score_dict = metricObject.update_running_average(model, batch, predictFunc)
      score = score_dict
      scoreList += [score]
      if score == 0:
        Corrects += [i]

      else:
        Wrongs += [i]

      progress = ("%d/%d - Validating Test set - %s: %.3f" % 
                 (i, n_batches, metric_name, 
                  metricObject.get_running_average()))

      if verbose==1:
        pbar.set_description(progress)
        pbar.update(1)

      elif verbose == 2:
        print(progress)

    if verbose==1:
      pbar.close()

    scores = np.array(scoreList)

    return {"score":metricObject.get_running_average(),
            "metric_name":metric_name, 
            "Corrects":Corrects, "Wrongs":Wrongs,
            "max_score":scores.max(), "min_score":scores.min(), 
            "mean_score":scores.mean(),
            "n_corrects":len(Corrects), "n_wrongs":len(Wrongs)}




# SCORERS
class AvgMeter:
    def __init__(self):
        self.dict = {}

    def __repr__(self):
        return self.get_string()

    def update(self, name, score, batch_size=None):
        if name not in self.dict:
          self.dict[name] = 0
          self.dict[name + "_n"] = 0

        if batch_size is None:
          batch_size = 1

        self.dict[name] += score
        self.dict[name + "_n"] += batch_size

    def get_dict(self):

      metricList = [m for m in self.dict if "_n" not in m]

      score = {}
      for m in metricList:
        num = self.dict[m]
        denom = self.dict[m + "_n"]

        if isinstance(num, np.ndarray):
          nz = denom != 0
          mscore = num.astype(np.float)

          mscore[nz] = mscore[nz] / denom[nz].astype(float) 

          score[m] = (mscore[nz].sum() / nz.sum())
          
        else:
          score[m] = num / denom

      return score

    def get_string(self):
        score_dict = self.get_dict()

        return dict2str(score_dict)

def dict2str(score_dict):
  string = ""
  for s in score_dict:
      string += " - %s: %.3f" % (s, score_dict[s])

  return string[3:]


