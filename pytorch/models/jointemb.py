from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as weight_init
from torch import optim
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from modules import SeqEncoder, BOWEncoder

class JointEmbeder(nn.Module):
    def __init__(self, config):
        super(JointEmbeder, self).__init__()
        self.conf = config
        self.margin = config['margin']
                
        self.name_encoder=SeqEncoder(config['n_words'],config['emb_size'],config['lstm_dims'])
        self.api_encoder=SeqEncoder(config['n_words'],config['emb_size'],config['lstm_dims'])
        self.tok_encoder=BOWEncoder(config['n_words'],config['emb_size'],config['n_hidden'])
        self.desc_encoder=SeqEncoder(config['n_words'],config['emb_size'],config['lstm_dims'])
        self.fuse=nn.Linear(config['emb_size']+4*config['lstm_dims'], config['n_hidden'])
        
        #create a model path to store model info
        if not os.path.exists(config['workdir']+'models/'):
            os.makedirs(config['workdir']+'models/')
    
    def code_encoding(self, name, api, tokens):
        name_repr=self.name_encoder(name)
        api_repr=self.api_encoder(api)
        tok_repr=self.tok_encoder(tokens)
        code_repr= self.fuse(torch.cat((name_repr, api_repr, tok_repr),1))
        code_repr=torch.tanh(code_repr)
        return code_repr
        
    def desc_encoding(self, desc):
        desc_repr=self.desc_encoder(desc)
        return desc_repr
    
    def forward(self, name, apiseq, tokens, desc_good, desc_bad): #self.data_params['methname_len']
        batch_size=name.size(0)
        code_repr=self.code_encoding(name, apiseq, tokens)
        desc_good_repr=self.desc_encoding(desc_good)
        desc_bad_repr=self.desc_encoding(desc_bad)
    
        good_sim=F.cosine_similarity(code_repr, desc_good_repr)
        bad_sim=F.cosine_similarity(code_repr, desc_bad_repr) # [batch_sz x 1]
        
        loss=(self.margin-good_sim+bad_sim).clamp(min=1e-6).mean()
        return loss