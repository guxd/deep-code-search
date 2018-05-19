from __future__ import print_function
from __future__ import absolute_import
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as weight_init
from torch import optim
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)
   
class BOWEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super(BOWEncoder, self).__init__()
        self.emb_size=emb_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        
    def forward(self, input, input_lengths=None): 
        batch_size, seq_len =input.size()
        embedded = self.embedding(input)  # input: [batch_sz x seq_len x 1]  embedded: [batch_sz x seq_len x emb_sz]
        embedded= F.dropout(embedded, 0.25, self.training) # [batch_size x seq_len x emb_size]
        output_pool = F.max_pool1d(embedded.transpose(1,2), seq_len).squeeze(2) # [batch_size x emb_size]
        encoding = F.tanh(output_pool)        
        return encoding
        
class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, n_layers=1):
        super(SeqEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, dropout=0.2, batch_first=True, bidirectional=True)
        for w in self.lstm.parameters(): # initialize the gate weights with orthogonal
            if w.dim()>1:
                weight_init.orthogonal_(w)

    def forward(self, input, input_lengths=None): 
        batch_size, seq_len=input.size()
        embedded = self.embedding(input)  # input: [batch_sz x seq_len]  embedded: [batch_sz x seq_len x emb_sz]
        embedded = F.dropout(embedded, 0.25, self.training)
        rnn_output, hidden = self.lstm(embedded) # out:[b x seq x hid_sz*2](biRNN) 
        rnn_output = F.dropout(rnn_output, 0.25, self.training)
        output_pool = F.max_pool1d(rnn_output.transpose(1,2), seq_len).squeeze(2) # [batch_size x hid_size*2]
        encoding = F.tanh(output_pool)

        return encoding
    
    
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
        code_repr=F.tanh(code_repr)
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


 
 
 
 