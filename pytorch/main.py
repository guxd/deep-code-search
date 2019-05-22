import os
import sys
import random
import traceback
import numpy as np
import math
from math import log
import argparse
from datashape.coretypes import real
random.seed(42)
import threading
import codecs
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

import torch
from torch import optim
import torch.nn.functional as F

from utils import cos_np, normalize, dot_np, sent2indexes
from configs import get_config
from data_loader import load_dict, CodeSearchDataset, load_vecs, save_vecs
import models

class CodeSearcher:
    def __init__(self, conf):
        self.conf=conf
        self.path = conf['workdir']
                
        self.vocab_methname = load_dict(self.path+conf['vocab_name'])
        self.vocab_apiseq=load_dict(self.path+conf['vocab_api'])
        self.vocab_tokens=load_dict(self.path+conf['vocab_tokens'])
        self.vocab_desc=load_dict(self.path+conf['vocab_desc'])
        
        self.codevecs=[]
        self.codebase= []
        self.codebase_chunksize=2000000
        
        self.valid_set = None
       

    ##### Data Set #####   
    def load_codebase(self):
        """load codebase
        codefile: h5 file that stores raw code
        """
        logger.info('Loading codebase (chunk size={})..'.format(self.codebase_chunksize))
        if not self.codebase: #empty
            codes=codecs.open(self.path+self.conf['use_codebase']).readlines()
                #use codecs to read in case of encoding problem
            for i in range(0,len(codes),self.codebase_chunksize):
                self.codebase.append(codes[i:i+self.codebase_chunksize])            
    
    ### Results Data ###
    def load_codevecs(self):
        logger.debug('Loading code vectors..')
        if not self.codevecs: # empty         
            """read vectors (2D numpy array) from a hdf5 file"""
            reprs=load_vecs(self.path+self.conf['use_codevecs'])
            for i in range(0,reprs.shape[0], self.codebase_chunksize):
                self.codevecs.append(reprs[i:i+self.codebase_chunksize])
       
       
    ##### Model Loading / saving #####
    def save_model(self, model, epoch):
        if not os.path.exists(self.path+'models/'):
            os.makedirs(self.path+'models/')
        torch.save(model.state_dict(), self.path+'models/epo%d.h5' % epoch)
        
    def load_model(self, model, epoch):
        assert os.path.exists(self.path+'models/epo%d.h5'%epoch), 'Weights at epoch %d not found' % epoch
        model.load_state_dict(torch.load(self.path+'models/epo%d.h5' % epoch))
        
        

    ##### Training #####
    def train(self, model):
        model.train()
        
        log_every = self.conf['log_every']
        valid_every = self.conf['valid_every']
        save_every = self.conf['save_every']
        batch_size = self.conf['batch_size']
        nb_epoch = self.conf['nb_epoch']
        
        train_set = CodeSearchDataset(self.path,
                                      self.conf['train_name'],self.conf['name_len'],
                                      self.conf['train_api'],self.conf['api_len'],
                                      self.conf['train_tokens'],self.conf['tokens_len'],
                                      self.conf['train_desc'],self.conf['desc_len'])
        
        data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.conf['batch_size'], 
                                           shuffle=True, drop_last=True, num_workers=1)
        
        val_loss = {'loss': 1., 'epoch': 0}

        for epoch in range(self.conf['reload']+1, nb_epoch):          
            itr = 1
            losses=[]
            for names, apis, toks, good_descs, bad_descs in data_loader:
                names, apis, toks, good_descs, bad_descs = [tensor.to(self.device) for tensor in [names, apis, toks, good_descs, bad_descs]]
                loss = model(names, apis, toks, good_descs, bad_descs)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if itr % log_every ==0:
                    logger.info('epo:[%d/%d] itr:%d Loss=%.5f'%(epoch, nb_epoch, itr, np.mean(losses)))
                    losses=[]
                itr = itr + 1    
            
      #      if epoch and epoch % valid_every == 0:
      #          logger.info("validating..")
      #          acc1, mrr, map, ndcg = self.eval(model,1000,1)              
                        
            if epoch and epoch % save_every == 0:
                self.save_model(model, epoch)

    ##### Evaluation #####
    def eval(self, model, poolsize, K):
        """
        simple validation in a code pool. 
        @param: poolsize - size of the code pool, if -1, load the whole test set
        """
        def ACC(real,predict):
            sum=0.0
            for val in real:
                try:
                    index=predict.index(val)
                except ValueError:
                    index=-1
                if index!=-1:
                    sum=sum+1  
            return sum/float(len(real))
        def MAP(real,predict):
            sum=0.0
            for id,val in enumerate(real):
                try:
                    index=predict.index(val)
                except ValueError:
                    index=-1
                if index!=-1:
                    sum=sum+(id+1)/float(index+1)
            return sum/float(len(real))
        def MRR(real,predict):
            sum=0.0
            for val in real:
                try:
                    index=predict.index(val)
                except ValueError:
                    index=-1
                if index!=-1:
                    sum=sum+1.0/float(index+1)
            return sum/float(len(real))
        def NDCG(real,predict):
            dcg=0.0
            idcg=IDCG(len(real))
            for i,predictItem in enumerate(predict):
                if predictItem in real:
                    itemRelevance=1
                    rank = i+1
                    dcg+=(math.pow(2,itemRelevance)-1.0)*(math.log(2)/math.log(rank+1))
            return dcg/float(idcg)
        def IDCG(n):
            idcg=0
            itemRelevance=1
            for i in range(n):
                idcg+=(math.pow(2,itemRelevance)-1.0)*(math.log(2)/math.log(i+2))
            return idcg

        if self.valid_set is None:        #load test dataset
            self.valid_set=CodeSearchDataset(self.path,
                                      self.conf['valid_name'],self.conf['name_len'],
                                      self.conf['valid_api'],self.conf['api_len'],
                                      self.conf['valid_tokens'],self.conf['tokens_len'],
                                      self.conf['valid_desc'],self.conf['desc_len'])
        
        data_loader = torch.utils.data.DataLoader(dataset=self.valid_set, batch_size=poolsize, 
                                           shuffle=True, drop_last=True, num_workers=1)
        model.eval()
        accs,mrrs,maps,ndcgs=[],[],[],[]
        for names, apis, toks, descs, _ in tqdm(data_loader):
            names, apis, toks, descs = [tensor.to(self.device) for tensor in [ names, apis, toks, descs]]
            code_repr=model.code_encoding(names, apis, toks)
            for i in range(poolsize):
                desc=descs[i].expand(poolsize,-1)
                desc_repr=model.desc_encoding(desc)
                n_results = K          
                sims = F.cosine_similarity(code_repr, desc_repr).data.cpu().numpy()
                negsims=np.negative(sims)
                predict=np.argsort(negsims)#predict = np.argpartition(negsims, kth=n_results-1)
                predict = predict[:n_results]   
                predict = [int(k) for k in predict]
                real=[i]
                accs.append(ACC(real,predict))
                mrrs.append(MRR(real,predict))
                maps.append(MAP(real,predict))
                ndcgs.append(NDCG(real,predict))                          
        logger.info('ACC={}, MRR={}, MAP={}, nDCG={}'.format(np.mean(accs),np.mean(mrrs),np.mean(maps),np.mean(ndcgs)))
        
        return np.mean(accs),np.mean(mrrs),np.mean(maps),np.mean(ndcgs)
    
    
    ##### Compute Representation #####
    def repr_code(self,model):
        model.eval()
        vecs=None
        use_set = CodeSearchDataset(self.conf['workdir'],
                                      self.conf['use_names'],self.conf['name_len'],
                                      self.conf['use_apis'],self.conf['api_len'],
                                      self.conf['use_tokens'],self.conf['tokens_len'])
        
        data_loader = torch.utils.data.DataLoader(dataset=use_set, batch_size=1000, 
                                           shuffle=False, drop_last=False, num_workers=1)
        for names,apis,toks in data_loader:
            names, apis, toks = [tensor.to(self.device) for tensor in [names, apis, toks]]
            reprs = model.code_encoding(names,apis,toks).data.cpu().numpy()
            vecs=reprs if vecs is None else np.concatenate((vecs, reprs),0)
        vecs = normalize(vecs)
        save_vecs(vecs,self.path+self.conf['use_codevecs'])
        return vecs
            
    
    def search(self,model,query,n_results=10):
        model.eval()
        desc=sent2indexes(query, self.vocab_desc)#convert desc sentence into word indices
        desc = np.expand_dims(desc, axis=0)
        desc= torch.from_numpy(desc).to(self.device)
        desc_repr=model.desc_encoding(desc).data.cpu().numpy()
        
        codes=[]
        sims=[]
        threads=[]
        for i, codevecs_chunk in enumerate(self.codevecs):
            t = threading.Thread(target=self.search_thread, args = (codes,sims,desc_repr, codevecs_chunk,i,n_results))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:#wait until all sub-threads finish
            t.join()
        return codes,sims
                 
    def search_thread(self,codes,sims,desc_repr,codevecs,i,n_results):        
    #1. compute code similarities
        chunk_sims=dot_np(normalize(desc_repr),codevecs) 
        
    #2. choose the top K results
        negsims=np.negative(chunk_sims[0])
        maxinds = np.argpartition(negsims, kth=n_results-1)
        maxinds = maxinds[:n_results]        
        chunk_codes=[self.codebase[i][k] for k in maxinds]
        chunk_sims=chunk_sims[0][maxinds]
        codes.extend(chunk_codes)
        sims.extend(chunk_sims)

    
def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search(Embedding) Model")
    parser.add_argument('--model', type=str, default='JointEmbeder', help='model name')
    parser.add_argument("--mode", choices=["train","eval","repr_code","search"], default='train',
                        help="The mode to run. The `train` mode trains a model;"
                        " the `eval` mode evaluat models in a test set "
                        " The `repr_code/repr_desc` mode computes vectors"
                        " for a code snippet or a natural language description with a trained model.")
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument("--verbose",action="store_true", default=True, help="Be verbose")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    conf = get_config()
    searcher = CodeSearcher(conf)
    searcher.device = device
    
    ##### Define model ######
    logger.info('Build Model')
    model = getattr(models, args.model)(conf)#initialize the model
    if conf['reload']>0:
        searcher.load_model(model, conf['reload'])
        
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=conf['lr'])
    
    if args.mode=='train':  
        searcher.train(model)
        
    elif args.mode=='eval':
        # evaluate for a particular epoch
        searcher.eval(model,1000,10)
        
    elif args.mode=='repr_code':
        vecs=searcher.repr_code(model)
        
    elif args.mode=='search':
        #search code based on a desc
        searcher.load_codevecs()
        searcher.load_codebase()
        while True:
            try:
                query = input('Input Query: ')
                n_results = int(input('How many results? '))
            except Exception:
                print("Exception while parsing your input:")
                traceback.print_exc()
                break
            codes,sims=searcher.search(model, query,n_results)
            zipped=zip(codes,sims)
            results = '\n\n'.join(map(str,zipped)) #combine the result into a returning string
            print(results)
