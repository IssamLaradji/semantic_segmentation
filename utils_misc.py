import json
import torch
import numpy as np
import subprocess
import json
import torch
import numpy as np
from tqdm import tqdm 
from torchvision import transforms
from torchvision.transforms import functional as ft
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import functional as ft
from importlib import reload
from skimage.segmentation import mark_boundaries
from torch.utils import data
import pickle 
import pandas as pd
from skimage import morphology as morph
import collections
import shlex
import inspect
from bs4 import BeautifulSoup
import tqdm
from torch.utils.data.dataloader import default_collate
import time 

def dict2frame(myDict):
  if len(myDict) == 0:
    return None 

  df=pd.DataFrame()

  for key in myDict:
    row = key[0]
    col = key[1]

    df.loc[row, col] = myDict[key]

  return df

def get_batch(datasets, indices):
  return default_collate([datasets[i] for i in indices])


import pprint
#import utils_main as mu

def argmax_mask(X, mask):
    ind_local = np.argmax(X[mask])

    G = np.ravel_multi_index(np.where(mask), mask.shape)
    Gi = np.unravel_index(G[ind_local], mask.shape)
    
    return Gi 

# def argmax_mask(X, mask):
#     ind = np.meshgrid(np.where(mask))
#     return np.argmax(X[ind])

# def up():
#     globals().update(locals())
def imsave(fname, arr):
    arr = f2l(t2n(arr)).squeeze()
    create_dirs(fname + "tmp")
    #print(arr.shape)
    scipy.misc.imsave(fname, arr)

def t2f(X):
    return Variable(torch.FloatTensor(X).cuda())

def t2l(X):
    return Variable(torch.LongTensor(X).cuda())
def get_size(model):
    total_size = 0
    for tensor in model.state_dict().values():
        total_size += tensor.numel() * tensor.element_size()
    return total_size / (1024.**3)

def ToPil(inputList):
    result = []

    for i in inputList:
        result += [transforms.functional.to_pil_image(i)]

    return result 

def point2mask(pointList, image, n_classes=None, return_count=False):
    h, w = np.asarray(image).shape[:2]
    points = np.zeros((h, w, 1), np.uint8)
    if return_count:
        counts = np.zeros(n_classes)

    for p in pointList: 
        if  int(p["x"]) > w or int(p["y"]) > h:
            continue
        else:
            points[int(p["y"]), int(p["x"])] = p["cls"]
            if return_count:
                counts[p["cls"]-1] += 1

    if return_count:
        return points, counts

    return points

def dict2frame(myDict):
  if len(myDict) == 0:
    return None 

  df=pd.DataFrame()

  for key in myDict:
    row = key[0]
    col = key[1]

    df.loc[row, col] = myDict[key]

  return df




def label2hot(y, n_classes):
    n = y.shape[0]
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), y] = 1

    return Y
    
def get_exp_name(dataset_name, config_name, main_dict, return_dict=False):
    name2id = {"metricList":"m"}

    keys2override = ["model_name","sampler_name",
                     "batch_size","opt_name","learning_rate","loss_name","weight_decay","epoch2val",
                     "iter2val", "epochs",
                     "dataset_options","metricList","model_options",
                     "trainTransformer","testTransformer",
                     "val_batchsize"]

    config = jload("configs.json")[config_name]
    config_args = parser_config.parse_config(config)
    config_dict = vars(config_args)

    exp_name = config_name + "-d:%s" % dataset_name

    value_dict = {}
    for key in keys2override:        
        if key in main_dict and main_dict[key] != None and main_dict[key] != config_dict[key]:
            value = main_dict[key]

            if isinstance(value, list):
                exp_name += "-%s:%s" % (name2id[key], value[0])
            elif key in ["epochs"]:
                pass
            else:
                exp_name += "-%s:%s" % (name2id[key], value)

        elif key in config_dict:
            value = config_dict[key]

        else:
            raise ValueError("%s does not exist..." % key) 

        value_dict[key] = value

    if return_dict:
        return exp_name, value_dict

    return exp_name
# import types
# def get_modules(module):
#     modules = {}

#     for name, val in module.__dict__.items():
#         if name in modules:
#           raise ValueError("Repeated module %s" % name) 
     
#         if isinstance(val, types.ModuleType):
#           modules[name] = val

    
#     return modules

    

def get_functions(module):
    funcs = {}
    for name, val in module.__dict__.items():
      if name in funcs:
        raise ValueError("Repeated func %s" % name) 
     

      if callable(val):
         funcs[name] = val

      
    return funcs
def old2new(path):
    return path.replace("/mnt/AIDATA/home/issam.laradji", 
                        "/mnt/home/issam")
def logsumexp(vals, dim=None):
    m = torch.max(vals, dim)[0]

    if dim is None:
        return m + torch.log(torch.sum(torch.exp(vals - m), dim))
    else:
        return m + torch.log(torch.sum(torch.exp(vals - m.unsqueeze(dim)), dim))
        
def count2weight(counts):
    uni, freq = np.unique(counts, return_counts=True)
    myDict = {i:j for i,j in zip(uni, freq)}
    freq = np.vectorize(myDict.get)(counts)

    return 1./freq
    
def time_elapsed(s_time):
    return (time.time() - s_time) / 60
    
def get_longest_list(listOfLists):
    LL = listOfLists
    longest_list = []

    if LL is None:
        return longest_list

    for L in LL:
        if not isinstance(L, list):
            continue

        if not isinstance(L[0], list):
            L = [L]
        
        if len(L) > len(longest_list):
            longest_list = L

    #print(longest_list)
    return longest_list

def n2l(A):
    return Variable(torch.LongTensor(A).cuda())
    
def get_median_list(listOfLists):
    LL = listOfLists
    pointList = []
    lenList = []

    if LL is None:
        return pointList

    for L in LL:
        if not isinstance(L, list):
            continue

        if not isinstance(L[0], list):
            L = [L]
        
        
        pointList += [L]
        lenList += [len(L)]
    if len(pointList) == 0:
        return pointList
        
    i = np.argsort(lenList)[len(lenList)//2]
    return pointList[i]



def get_histogram(dataset):
    n = len(dataset)
    n_classes = t2n(dataset[0]["counts"]).size
    
    counts = np.zeros((n, n_classes))
    pbar = tqdm.tqdm(total=len(dataset), leave=False)
    for i in range(len(dataset)):
        counts[i] = t2n(dataset[i]["counts"])

        pbar.update(1)
    pbar.close()
    return counts

def count2stats(countStats):
    pass

def shrink2roi(img, roi):
    ind = np.where(roi != 0)

    y_min = min(ind[0])
    y_max = max(ind[0])

    x_min = min(ind[1])
    x_max = max(ind[1])

    return img[y_min:y_max, x_min:x_max]


  
def read_xml(fname):
    with open(fname) as f:
        xml = f.readlines()
        xml = ''.join([line.strip('\t') for line in xml])
        
        xml = BeautifulSoup(xml, "lxml")

    return xml

def getFileFunctions(fname):
    name_func_tuples = inspect.getmembers(fname, inspect.isfunction)
    name_func_tuples = [t for t in name_func_tuples if inspect.getmodule(t[1]) == fname]
    functions = dict(name_func_tuples)

    return functions 

def add2diag(A, eps=1e-6):
    n = A.size(0)
    if A.is_cuda:
        return A + Variable(torch.eye(n).cuda()*eps)
    else:
        return A + Variable(torch.eye(n)*eps)

def batch_tril(A):
    B = A.clone()
    ii,jj = np.triu_indices(B.size(-2), k=1, m=B.size(-1))
    B[...,ii,jj] = 0
    return B
    
def batch_diag(A):
    ii,jj = np.diag_indices(min(A.size(-2),A.size(-1)))
    return A[...,ii,jj]




def unique(tensor, return_counts=0):
    return np.unique(t2n(tensor), return_counts=return_counts)

def read_text(fname):
    # READS LINES
    with open(fname, "r") as f:
        lines = f.readlines()
    return lines

def read_textraw(fname):
    with open(fname, "r") as f:
        lines = f.read()
    return lines   

def parse_command(command, parser):    
    if isinstance(command, list):
      command = " ".join(command)
      
    io_args = parser.parse_args(shlex.split(command))

    return io_args

def dict2dataframe(dicts, on):
    names = list(dicts.keys()) 
    trh = pd.DataFrame(dicts[names[0]])
    teh = pd.DataFrame(dicts[names[1]])
    df = pd.merge(trh, teh, on=on, how="outer", sort=on, suffixes=("_%s" % names[0],
                                             "_%s" % names[1]))

    return df
def extract_fname(directory):
    import ntpath
    return ntpath.basename(directory)


def dict2name(my_dict):
    new_dict = collections.OrderedDict(sorted(my_dict.items()))

    name = "_".join(map(str, list(new_dict.values())))

    return name


def gray2cmap(gray, cmap="jet", thresh=0):
    # Gray has values between 0 and 255 or 0 and 1
    gray = t2n(gray)
    gray = gray / gray.max()
    gray = np.maximum(gray - thresh, 0)
    gray = gray / gray.max()
    gray = gray * 255

    gray = gray.astype(int)
    #print(gray)
   
    from pylab import get_cmap
    cmap = get_cmap(cmap)

    output = np.zeros(gray.shape + (3,), dtype=np.float64)

    for c in np.unique(gray):
        output[(gray==c).nonzero()] = cmap(c)[:3]

    return l2f(output)


import PIL

def n2p(img):
    im = PIL.Image.fromarray(np.uint8(img*255))

    return im

def get_counts():
    pass

def create_dirs(fname):
    if "/" not in fname:
        return
        
    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass 
            
            
def save_pkl(fname, dict):
    create_dirs(fname)
    with open(fname, "wb") as f: 
        pickle.dump(dict, f)

def jload(fname):
    with open(fname) as data_file:
        return json.loads(data_file.read())

def load_pkl(fname):
    with open(fname, "rb") as f:        
        return pickle.load(f)


def label2Image(imgs):
    imgs = t2n(imgs).copy()

    if imgs.ndim == 3:
        imgs = imgs[:, np.newaxis]

    imgs = l2f(imgs)

    if imgs.ndim == 4 and imgs.shape[1] != 1:
        imgs = np.argmax(imgs, 1)

    imgs = label2rgb(imgs)

    if imgs.ndim == 3:
        imgs = imgs[np.newaxis]
    return imgs

def run_bash_command(command, noSplit=True):
    if noSplit:
        command = command.split()
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = process.communicate()

    return str(output)

def run_bash(fname, arg1):
    return subprocess.check_call([fname, arg1])

# def label2Image(imgs, win="8888", nrow=4):
#     # If given a single image
#     imgs = t2n(imgs).copy()

#     # Label image case
#     if imgs.ndim == 2:
#         imgs = mask2label(imgs)
#         imgs = l2f(imgs)

#     # Prediction output case
#     if imgs.ndim == 4:
#         imgs = np.argmax(imgs, 1)
    
#     imgs = label2rgb(imgs, np.max(np.unique(imgs)) + 1)


#     return imgs

def create_dirs(fname):
    if "/" not in fname:
        return
        
    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass 
            
def stack(imgList):
    imgListNumpy = []
    for img in imgList:
        new_img = l2f(t2n(img)).copy()
        if new_img.max() > 1:
            new_img = new_img / 255.

        imgListNumpy += [new_img]

    return np.vstack(imgListNumpy)

def maskOnImage(imgs, mask, enlarge=0):
    imgs = l2f(t2n(imgs)).copy()
    mask = label2Image(mask)
    
    if enlarge:
        mask = zoom(mask, 11)

    if mask.max() > 1:
        mask = mask / 255.

    if imgs.max() > 1:
        imgs = imgs / 255.
    
    nz = mask != 0 
    imgs = imgs*0.5 + mask * 0.5
    imgs = imgs/imgs.max()
    #print(imgs[nz])
    #print(imgs.shape)
    #print(mask.shape)
    if mask.ndim == 4:
        mask = mask.sum(1)

    nz = mask != 0
    mask[nz] = 1

    mask = mask.astype(int)

    #imgs = imgs*0.5 + mask[:, :, :, np.newaxis] * 0.5

    segList = []
    for i in range(imgs.shape[0]):
        segList += [l2f(mark_boundaries(f2l(imgs[i]).copy(), f2l(mask[i]),mode="outer"))]
    
    imgs = np.stack(segList)

    return l2f(imgs)

def labelrgb2label(labels):
    gray_label = np.zeros((labels.shape[0], labels.shape[1]), dtype=np.uint8)

    rgbs = {(0,0,0):0}
    c_id = 1
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            c = tuple(labels[i,j])
            if c not in rgbs:
                rgbs[c] = c_id
                c_id += 1
            

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            c = tuple(labels[i,j])
            gray_label[i, j] = rgbs[c]
    

    return gray_label



def rgb2label(img, n_classes, void_class=-1):
    rgb = img.copy()

    label = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)


    
    classes = np.arange(n_classes).tolist()

    # if void is not None:
    #     N = max(n_classes, void) + 1
    #     classes += [void]
    # else:
    N = n_classes + 1

    colors = color_map(N=N)

    for c in classes:
        label[np.where(np.all(rgb == colors[c], axis=-1))[:2]] = c

    # label[np.where(np.all(rgb == colors[c], axis=-1))[:2]] = c
    
    return label

def label2rgb(labels, bglabel=None, bg_color=(0., 0., 0.)):
    labels = np.squeeze(labels)
    colors = color_map(np.max(np.unique(labels)) + 1)
    output = np.zeros(labels.shape + (3,), dtype=np.float64)

    for i in range(len(colors)):
        if i != bglabel:
            output[(labels == i).nonzero()] = colors[i]

    if bglabel is not None:
        output[(labels == bglabel).nonzero()] = bg_color

    return l2f(output)

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def zoom(img,kernel_size=3):
    img = n2t(img)
    if img.dim() == 4:
        img = img.sum(1).unsqueeze(1)
    img = Variable(n2t(img)).float()
    img = F.max_pool2d(img, kernel_size=kernel_size, stride=1, 
                       padding=get_padding(kernel_size))
    return t2n(img)

def numpy2seq(Z, val=-1):
    """Appends the minimal required amount of zeroes at the end of each 
     array in the jagged array `M`, such that `M` looses its jagedness."""
    seq = []
    for z in t2n(Z).astype(int):
        i = np.where(z==val)[0]
        if i.size == 0:
            seq += [z.tolist()]
        else:
            seq += [z[:min(i)].tolist()]
        
    return seq

def seq2numpy(M, val=-1, maxlen=None):
    """Appends the minimal required amount of zeroes at the end of each 
     array in the jagged array `M`, such that `M` looses its jagedness."""
    if maxlen is None:
        maxlen = max(len(r) for r in M)

    Z = np.ones((len(M), maxlen)) * val
    for i, row in enumerate(M):
        Z[i, :len(row)] = row 
        
    return Z

def get_padding(kernel_size=1):
    return int((kernel_size - 1) / 2)

# MISC
def remove_dir(dir_name):
    import shutil
    shutil.rmtree(dir_name)

def dict2str(score):
    string = ""
    for k in score:
        string += "- %s - %.3f" % (k, score[k])
    return string[2:]

def save_csv(fname, df):
    create_dirs(fname)
    df.to_csv(fname, index=False)

def save_json(fname, data):
    create_dirs(fname)
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)
    
def load_json(fname):
    with open(fname, "r") as json_file:
        d = json.load(json_file)
    
    return d

def print_box(*strings):
    string_format = ["{%d:10s}" % i for i in range(len(strings))]
    

    sizes = [len(i) for i in strings]
    bars = ["-"*s for s in sizes]
    print("\n")
    print(" ".join(string_format).format(*bars))
    print(" ".join(string_format).format(*strings))
    print(" ".join(string_format).format(*bars))

def print_header(*strings):
    string_format = ["{%d:10s}" % i for i in range(len(strings))]
    print("\n"+" ".join(string_format).format(*strings))

    sizes = [len(i) for i in strings]
    bars = ["-"*s for s in sizes]
    print(" ".join(string_format).format(*bars))

def pprint(*strings):
    string_format = ["{%d:10s}" % i for i in range(len(strings))]
    #string_format[0] = "{%d:5s}"
    strings = [str(s) for s in strings]
    print(" ".join(string_format).format(*strings))


def f2l(X):
    if X.ndim == 3 and (X.shape[2] == 3 or X.shape[2] == 1):
        return X
    if X.ndim == 4 and (X.shape[3] == 3 or X.shape[3] == 1):
        return X

    # CHANNELS FIRST
    if X.ndim == 3:
        return np.transpose(X, (1,2,0))
    if X.ndim == 4:
        return np.transpose(X, (0,2,3,1))

    return X

def l2f(X):
    if X.ndim == 3 and (X.shape[0] == 3 or X.shape[0] == 1):
        return X
    if X.ndim == 4 and (X.shape[1] == 3 or X.shape[1] == 1):
        return X

    if X.ndim == 4 and (X.shape[1] < X.shape[3]):
        return X

    # CHANNELS LAST
    if X.ndim == 3:
        return np.transpose(X, (2,0,1))
    if X.ndim == 4:
        return np.transpose(X, (0,3,1,2))

    return X


def stack_images(images):
    for img in images:
        import ipdb; ipdb.set_trace()  # breakpoint f1a9702d //
        

def t2n(x):
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, torch.autograd.Variable):
        x = x.cpu().data.numpy()
        
    if isinstance(x, (torch.cuda.FloatTensor, torch.cuda.IntTensor, torch.cuda.LongTensor, torch.cuda.DoubleTensor )):
        x = x.cpu().numpy()

    if isinstance(x, (torch.FloatTensor, torch.IntTensor, torch.LongTensor, torch.DoubleTensor )):
        x = x.numpy()

    return x

def n2t(x, dtype="float"):
    if isinstance(x, (int, np.int64, float)):
        x = np.array([x])

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x

def n2v(x, dtype="float", cuda=True):
    if isinstance(x, (int, np.int64, float)):
        x = np.array([x])

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
      

    if isinstance(x, Variable):
        return x 

    if cuda:
        x = x.cuda()

    return Variable(x).float()




def print_config(configs):
    print("\n")
    pprint("dataset: %s" % configs["dataset"], "model: %s" % configs["model"], 
              "optimizer: %s" % configs["opt"])
    print("\n")


def zscale(X, mu=None, var=None, with_stats=False):
    if mu is None:
        mu = X.mean(0)
    if var is None:
        var = X.var(0)
    Xs =  (X - mu) / var
  
    if with_stats:
        return Xs, mu, var

    else:
        return Xs
#### TRAINERS



import scipy.misc
import scipy.io as io
import os 

def imread(fname):
    return scipy.misc.imread(fname)


def loadmat(fname):
    return io.loadmat(fname)

def count_files(dir):
    list = os.listdir(dir) 
    return len(list)

def f2n( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
      
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data

def create_latex(fname, title, desc, sectionList, figList):
    template=("\documentclass[12pt,a4paper]{article} % din a4",
        ", 11 pt, one sided\n\n",
    "\begin{document}\n",
    "\VAR{sec}\n",
    "\VAR{fig}\n")

    for i in range(len(sectionList)):
        template += "\n%s\n" % sectionList[i]
        template += "\n%s\n" % create_latex_fig(figList[i])

    template += "\end{document}"

    save_txt(fname, template)

def save_txt(fname, string):
    with open(fname, "w") as f:
        f.write(string)


def create_latex_fig(fname, img):
   
    imsave(fname, img)

    fig = ("\begin{figure}\n",
    "\centering\n", 
    "\includegraphics[width=4in]{%s}\n",
    "\end{figure}\n" % (fname))

    return fig


def create_latex_table(fname, df):

    fig = ("\begin{figure}\n",
    "\centering\n", 
    "\includegraphics[width=4in]{%s}\n",
    "\end{figure}\n" % (fname))

    return fig





def get_modules(module_name):
  import inspect
  import datasets, models
  from importlib import import_module
  
  mod_dict = {}

  modList = [import_module("{}.{}".format(module_name,
            class_name)) for class_name in 
            eval("%s.__all__"%module_name)]

  for module in modList:

    funcs = get_functions(module)
    for name in funcs:
      val = funcs[name]

      if not inspect.isclass(val):
        continue


      if (name in mod_dict and 
         module_name in str(val.__module__)):
         if name != "Pascal2012":
            raise ValueError("repeated %s" % name)
         print("Repeated:", name)
      mod_dict[name] = val

  return mod_dict
