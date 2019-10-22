import os
import sys
from datetime import datetime
import numpy as np
import argparse
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

import torch
from utils import normalize
from data_loader import CodeSearchDataset, save_vecs
import models, configs    

##### Compute Representation #####
def repr_code(args):

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    config=getattr(configs, 'config_'+args.model)()

    ##### Define model ######
    logger.info('Constructing Model..')
    model = getattr(models, args.model)(config)#initialize the model
    if args.reload_from>0:
        model.load_state_dict(torch.load(f'./output/{args.model}/{args.dataset}/models/epo{args.reload_from}.h5'))       
    model = model.to(device)   
    model.eval()

    data_path = args.data_path+args.dataset+'/'
    use_set = eval(config['dataset_name'])(data_path, config['use_names'], config['name_len'],
                              config['use_apis'], config['api_len'],
                              config['use_tokens'], config['tokens_len'])

    data_loader = torch.utils.data.DataLoader(dataset=use_set, batch_size=1000, 
                                  shuffle=False, drop_last=False, num_workers=1)

    vecs=None        
    for batch in tqdm(data_loader):
        batch_gpu = [tensor.to(device) for tensor in batch]
        with torch.no_grad():
            reprs = model.code_encoding(*batch_gpu).data.cpu().numpy()
        vecs=reprs if vecs is None else np.concatenate((vecs, reprs),0)
    vecs = normalize(vecs)
    save_vecs(vecs, data_path + config['use_codevecs'])
    
def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search(Embedding) Model")
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='JointEmbeder', help='model name')
    parser.add_argument('-d', '--dataset', type=str, default='github', help='dataset')
    parser.add_argument('--reload_from', type=int, default=-1, help='epoch to reload from')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-v', "--visual",action="store_true", default=False, help="Visualize training status in tensorboard")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    repr_code(args)

        
   