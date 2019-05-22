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
        encoding = torch.tanh(output_pool)        
        return encoding
        
class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, n_layers=1):
        super(SeqEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
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
        encoding = torch.tanh(output_pool)

        return encoding
    
    



 
 
 
 