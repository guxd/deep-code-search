import os
import sys
import traceback
import numpy as np
import argparse
import threading
import codecs
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
from tensorboardX import SummaryWriter # install tensorboardX (pip install tensorboardX) before importing this package

import torch

from utils import cos_np, normalize, dot_np, sent2indexes
from data_loader import load_dict, load_vecs
import models, configs
  
codevecs, codebase = [], []

##### Data Set #####   
def load_codebase(code_path, chunk_size=2000000):
    """load codebase
      codefile: h5 file that stores raw code
    """
    logger.info(f'Loading codebase (chunk size={chunk_size})..')
    codebase= []
    codes=codecs.open(code_path).readlines()
        #use codecs to read in case of encoding problem
    for i in range(0,len(codes), chunk_size):
        codebase.append(codes[i:i+chunk_size]) 
    return codebase

### Results Data ###
def load_codevecs(vec_path, chunk_size=2000000):
    logger.debug('Loading code vectors..')       
    """read vectors (2D numpy array) from a hdf5 file"""
    codevecs=[]
    reprs=load_vecs(vec_path)
    for i in range(0,reprs.shape[0], chunk_size):
        codevecs.append(reprs[i:i+chunk_size])
    return codevecs

def search(config, model, vocab, query, n_results=10):
    model.eval()
    device = next(model.parameters()).device
    desc, desc_len =sent2indexes(query, vocab_desc, config['desc_len'])#convert query into word indices
    desc= torch.from_numpy(desc).unsqueeze(0).to(device)
    desc_len = torch.zeros(1, dtype=torch.long, device=device).fill_(desc_len).clamp(max=config['desc_len'])
    with torch.no_grad():
        desc_repr=model.desc_encoding(desc, desc_len).data.cpu().numpy()

    codes, sims =[], []
    threads=[]
    for i, codevecs_chunk in enumerate(codevecs):
        t = threading.Thread(target=search_thread, args = (codes, sims, desc_repr, codevecs_chunk, i, n_results))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:#wait until all sub-threads have completed
        t.join()
    return codes,sims

def search_thread(codes, sims, desc_repr, codevecs, i, n_results):        
#1. compute code similarities
    chunk_sims=dot_np(normalize(desc_repr),codevecs) 
    chunk_sims = chunk_sims[0] # squeeze batch dim

#2. select the top K results
    negsims = np.negative(chunk_sims)
    maxinds = np.argpartition(negsims, kth=n_results-1)
    maxinds = maxinds[:n_results]        
    chunk_codes=[codebase[i][k] for k in maxinds]
    chunk_sims=chunk_sims[maxinds]
    codes.extend(chunk_codes)
    sims.extend(chunk_sims)
    
def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search(Embedding) Model")
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='JointEmbeder', help='model name')
    parser.add_argument('-d', '--dataset', type=str, default='github', help='name of dataset.java, python')
    parser.add_argument('--reload_from', type=int, default=-1, help='epoch to reload from')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-v', "--visual",action="store_true", default=False, help="Visualize training status in tensorboard")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    config=getattr(configs, 'config_'+args.model)()
    
    ##### Define model ######
    logger.info('Constructing Model..')
    model = getattr(models, args.model)(config)#initialize the model
    ckpt=f'./output/{args.model}/{args.dataset}/models/epo{args.reload_from}.h5'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    
    data_path = args.data_path+args.dataset+'/'
    
    vocab_desc=load_dict(data_path+config['vocab_desc'])
    #search code based on a desc
    codevecs = load_codevecs(data_path+config['use_codevecs'])
    codebase = load_codebase(data_path+config['use_codebase'])
        
    while True:
        try:
            query = input('Input Query: ')
            n_results = int(input('How many results? '))
        except Exception:
            print("Exception while parsing your input:")
            traceback.print_exc()
            break
        query = query.lower().replace('how to ', '').replace('how do i ', '').replace('how can i ', '').replace('?', '').strip()
        codes,sims = search(config, model, vocab_desc, query, n_results)
        zipped=zip(codes, sims)
        results = '\n\n'.join(map(str,zipped)) #combine the result into a returning string
        print(results)
