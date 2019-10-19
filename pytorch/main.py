import os
import sys
import random
import traceback
from datetime import datetime
import numpy as np
import math
import argparse
from datashape.coretypes import real
random.seed(42)
import threading
import codecs
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
from tensorboardX import SummaryWriter # install tensorboardX (pip install tensorboardX) before importing this package

import torch
from torch import optim
import torch.nn.functional as F

from utils import cos_np, normalize, dot_np, sent2indexes
from data_loader import load_dict, CodeSearchDataset, load_vecs, save_vecs
import models, configs

class CodeSearcher:
    def __init__(self, config):        
        self.vocab_methname = load_dict(config['data_path']+config['vocab_name'])
        self.vocab_apiseq=load_dict(config['data_path']+config['vocab_api'])
        self.vocab_tokens=load_dict(config['data_path']+config['vocab_tokens'])
        self.vocab_desc=load_dict(config['data_path']+config['vocab_desc'])
        
        self.codevecs= []
        self.codebase= []
        
        self.valid_set = None
       
    ##### Data Set #####   
    def load_codebase(self, code_path, chunk_size=2000000):
        """load codebase
          codefile: h5 file that stores raw code
        """
        logger.info(f'Loading codebase (chunk size={chunk_size})..')
        codes=codecs.open(code_path).readlines()
            #use codecs to read in case of encoding problem
        for i in range(0,len(codes), chunk_size):
            self.codebase.append(codes[i:i+chunk_size])            
    
    ### Results Data ###
    def load_codevecs(self, vec_path, chunk_size=2000000):
        logger.debug('Loading code vectors..')       
        """read vectors (2D numpy array) from a hdf5 file"""
        reprs=load_vecs(vec_path)
        for i in range(0,reprs.shape[0], chunk_size):
            self.codevecs.append(reprs[i:i+chunk_size])
       
       
    ##### Model Loading / saving #####
    def save_model(self, model, epoch):
        model_name = model.__class__.__name__
        torch.save(model.state_dict(), f'./output/{model_name}/epo{epoch}.h5')
        
    def load_model(self, model, epoch):
        model_name = model.__class__.__name__
        assert os.path.exists(f'./output/{model_name}/epo{epoch}.h5'), f'Weights at epoch {epoch} not found' % epoch
        model.load_state_dict(torch.load(f'./output/{model_name}/epo{epoch}.h5'))
        
        

    ##### Training #####
    def train(self, config, model, tb_writer):
        model.train()
        torch.backends.cudnn.benchmark = True # speed up training by using cudnn
        torch.backends.cudnn.deterministic = True # fix the random seed in cudnn        
        
        train_set = eval(config['dataset_name'])(config['data_path'], config['train_name'], config['name_len'],
                                      config['train_api'], config['api_len'],
                                      config['train_tokens'], config['tokens_len'],
                                      config['train_desc'], config['desc_len'])
        
        data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'], 
                                           shuffle=True, drop_last=True, num_workers=1)
        
        val_loss = {'loss': 1., 'epoch': 0}
        n_iters = len(data_loader)

        itr_global = config['reload']+1
        for epoch in range(int(config['reload']/n_iters)+1, config['nb_epoch']):         
            losses=[]
            for batch in data_loader:
                batch_gpu = [tensor.to(self.device) for tensor in batch]
                loss = model.train_batch(*batch_gpu)
                losses.append(loss)
                if itr_global % config['log_every'] ==0:
                    logger.info('epo:[%d/%d] itr:[%d/%d] Loss=%.5f'%
                                (epoch, config['nb_epoch'], itr_global%n_iters, n_iters, np.mean(losses)))
                    if tb_writer is not None:
                        tb_writer.add_scalar('loss', np.mean(losses), itr_global)
                    losses=[] 
                itr_global = itr_global + 1
            
                if itr_global % config['valid_every'] == 0:
                    logger.info("validating..")
                    acc1, mrr, map, ndcg = self.eval(config, model,1000,1)             
                        
                if itr_global % config['save_every'] == 0:
                    self.save_model(model, itr_global)

    ##### Evaluation #####
    def eval(self, config, model, poolsize, K):
        """
        simple validation in a code pool. 
        @param: poolsize - size of the code pool, if -1, load the whole test set
        """
        def ACC(real,predict):
            sum=0.0
            for val in real:
                try: index=predict.index(val)
                except ValueError: index=-1
                if index!=-1: sum=sum+1  
            return sum/float(len(real))
        def MAP(real,predict):
            sum=0.0
            for id,val in enumerate(real):
                try: index=predict.index(val)
                except ValueError: index=-1
                if index!=-1: sum=sum+(id+1)/float(index+1)
            return sum/float(len(real))
        def MRR(real,predict):
            sum=0.0
            for val in real:
                try: index=predict.index(val)
                except ValueError: index=-1
                if index!=-1: sum=sum+1.0/float(index+1)
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
            for i in range(n): idcg+=(math.pow(2,itemRelevance)-1.0)*(math.log(2)/math.log(i+2))
            return idcg
        
        model.eval()
        
        if self.valid_set is None:     #load test dataset
            self.valid_set=eval(config['dataset_name'])(config['data_path'],
                                      config['valid_name'], config['name_len'],
                                      config['valid_api'], config['api_len'],
                                      config['valid_tokens'], config['tokens_len'],
                                      config['valid_desc'], config['desc_len'])
        
        data_loader = torch.utils.data.DataLoader(dataset=self.valid_set, batch_size=poolsize, 
                                     shuffle=True, drop_last=True, num_workers=1)
        
        accs, mrrs, maps, ndcgs=[],[],[],[]
        for batch in tqdm(data_loader):
            code_batch = [tensor.to(self.device) for tensor in batch[:3]]
            desc_batch = [tensor.to(self.device) for tensor in batch[3:4]]
            with torch.no_grad():
                code_repr=model.code_encoding(*code_batch)
                desc_repr=model.desc_encoding(*desc_batch) # [poolsize x hid_size]
            for i in range(poolsize):
                desc_repr_rep=desc_repr[i].view(1, -1).expand(poolsize,-1)
                n_results = K          
                sims = F.cosine_similarity(code_repr, desc_repr_rep).data.cpu().numpy()
                negsims=np.negative(sims)
                predict=np.argsort(negsims)#predict = np.argpartition(negsims, kth=n_results-1)
                predict = predict[:n_results]   
                predict = [int(k) for k in predict]
                real=[i]
                accs.append(ACC(real,predict))
                mrrs.append(MRR(real,predict))
                maps.append(MAP(real,predict))
                ndcgs.append(NDCG(real,predict))                          
        logger.info(f'ACC={np.mean(accs)}, MRR={np.mean(mrrs)}, MAP={np.mean(maps)}, nDCG={np.mean(ndcgs)}')
        
        return np.mean(accs),np.mean(mrrs),np.mean(maps),np.mean(ndcgs)
    
    
    ##### Compute Representation #####
    def repr_code(self, config, model):
        model.eval()
        
        use_set = eval(config['dataset_name'])(config['data_path'], config['use_names'], config['name_len'],
                                  config['use_apis'], config['api_len'],
                                  config['use_tokens'], config['tokens_len'])
        
        data_loader = torch.utils.data.DataLoader(dataset=use_set, batch_size=1000, 
                                      shuffle=False, drop_last=False, num_workers=1)
        
        vecs=None        
        for names,apis,toks in tqdm(data_loader):
            names, apis, toks = [tensor.to(self.device) for tensor in [names, apis, toks]]
            with torch.no_grad():
                reprs = model.code_encoding(names,apis,toks).data.cpu().numpy()
            vecs=reprs if vecs is None else np.concatenate((vecs, reprs),0)
        vecs = normalize(vecs)
        save_vecs(vecs, config['data_path'] + config['use_codevecs'])
        return vecs
            
    
    def search(self, config, model,query,n_results=10):
        model.eval()
        desc=sent2indexes(query, self.vocab_desc, config['desc_len'])#convert query into word indices
        desc = np.expand_dims(desc, axis=0)
        desc= torch.from_numpy(desc).to(self.device)
        with torch.no_grad():
            desc_repr=model.desc_encoding(desc).data.cpu().numpy()
        
        codes, sims =[], []
        threads=[]
        for i, codevecs_chunk in enumerate(self.codevecs):
            t = threading.Thread(target=self.search_thread, args = (codes, sims, desc_repr, codevecs_chunk, i, n_results))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:#wait until all sub-threads have completed
            t.join()
        return codes,sims
                 
    def search_thread(self,codes,sims,desc_repr,codevecs,i,n_results):        
    #1. compute code similarities
        chunk_sims=dot_np(normalize(desc_repr),codevecs) 
        chunk_sims = chunk_sims[0] # squeeze batch dim
        
    #2. select the top K results
        negsims=np.negative(chunk_sims)
        maxinds = np.argpartition(negsims, kth=n_results-1)
        maxinds = maxinds[:n_results]        
        chunk_codes=[self.codebase[i][k] for k in maxinds]
        chunk_sims=chunk_sims[maxinds]
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
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-v', "--visual",action="store_true", default=True, help="Visualize training status in tensorboard")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    config=getattr(configs, 'config_'+args.model)()
    searcher = CodeSearcher(config)
    searcher.device = device
    
    ##### Define model ######
    logger.info('Constructing Model..')
    model = getattr(models, args.model)(config)#initialize the model
    if config['reload']>0:
        searcher.load_model(model, config['reload'])
        
    model = model.to(device)
    
    if args.mode=='train': 
        os.makedirs(f'./output/{args.model}/', exist_ok=True)
        if args.visual:
            timestamp = datetime.now().strftime('%Y%m%d%H%M')     
            os.makedirs(f"/output/{args.model}/logs/{timestamp}" , exist_ok=True)
            tb_writer = SummaryWriter(f"/output/{args.model}/logs/{timestamp}" )
        else: tb_writer = None
        searcher.train(config, model, tb_writer)
        
    elif args.mode=='eval':
        # evaluate for a particular epoch
        searcher.eval(config, model,1000,10)
        
    elif args.mode=='repr_code':
        vecs=searcher.repr_code(config, model)
        
    elif args.mode=='search':
        #search code based on a desc
        if not searcher.codevecs:
            searcher.load_codevecs(config['data_path']+config['use_codevecs'])
        if not searcher.codebase: #empty
            searcher.load_codebase(config['data_path']+config['use_codebase'])
        while True:
            try:
                query = input('Input Query: ')
                query = query.lower().replace('how do i ','').replace('how can i ','').replace('how to ','').strip()
                n_results = int(input('How many results? '))
            except Exception:
                print("Exception while parsing your input:")
                traceback.print_exc()
                break
            codes,sims=searcher.search(config, model, query, n_results)
            zipped=zip(codes, sims)
            results = '\n\n'.join(map(str,zipped)) #combine the result into a returning string
            print(results)
