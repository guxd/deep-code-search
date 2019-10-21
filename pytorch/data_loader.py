import sys
import torch 
import torch.utils.data as data
import torch.nn as nn
import tables
import json
import random
import numpy as np
import pickle
from utils import PAD_ID, SOS_ID, EOS_ID, UNK_ID 

    
class CodeSearchDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """
    def __init__(self, data_dir, f_name, max_name_len, f_api, max_api_len, 
                 f_tokens, max_tok_len, f_descs=None, max_desc_len=None):
        self.max_name_len=max_name_len
        self.max_api_len=max_api_len
        self.max_tok_len=max_tok_len
        self.max_desc_len=max_desc_len
        # 1. Initialize file path or list of file names.
        """read training data(list of int arrays) from a hdf5 file"""
        self.training=False
        print("loading data...")
        table_name = tables.open_file(data_dir+f_name)
        self.names = table_name.get_node('/phrases')
        self.idx_names = table_name.get_node('/indices')
        table_api = tables.open_file(data_dir+f_api)
        self.apis = table_api.get_node('/phrases')
        self.idx_apis = table_api.get_node('/indices')
        table_tokens = tables.open_file(data_dir+f_tokens)
        self.tokens = table_tokens.get_node('/phrases')
        self.idx_tokens = table_tokens.get_node('/indices')
        if f_descs is not None:
            self.training=True
            table_desc = tables.open_file(data_dir+f_descs)
            self.descs = table_desc.get_node('/phrases')
            self.idx_descs = table_desc.get_node('/indices')
        
        assert self.idx_names.shape[0] == self.idx_apis.shape[0]
        assert self.idx_apis.shape[0] == self.idx_tokens.shape[0]
        if f_descs is not None:
            assert self.idx_names.shape[0]==self.idx_descs.shape[0]
        self.data_len = self.idx_names.shape[0]
        print("{} entries".format(self.data_len))
        
    def pad_seq(self, seq, maxlen):
        if len(seq)<maxlen:
            seq=np.append(seq, [PAD_ID]*maxlen)
        seq=seq[:maxlen]
        return seq
    
    def __getitem__(self, offset):          
        len, pos = self.idx_names[offset]['length'], self.idx_names[offset]['pos']
        name = self.names[pos:pos + len].astype('int64')
        name = self.pad_seq(name, self.max_name_len)
        name_len=min(int(len),self.max_name_len) 
        
        len, pos = self.idx_apis[offset]['length'], self.idx_apis[offset]['pos']
        apiseq = self.apis[pos:pos+len].astype('int64')
        apiseq = self.pad_seq(apiseq, self.max_api_len)
        api_len = min(int(len), self.max_api_len)

        len, pos = self.idx_tokens[offset]['length'], self.idx_tokens[offset]['pos']
        tokens = self.tokens[pos:pos+len].astype('int64')
        tokens = self.pad_seq(tokens, self.max_tok_len)
        tok_len = min(int(len), self.max_tok_len)

        if self.training:
            len, pos = self.idx_descs[offset]['length'], self.idx_descs[offset]['pos']
            good_desc = self.descs[pos:pos+len].astype('int64')
            good_desc = self.pad_seq(good_desc, self.max_desc_len)
            good_desc_len = min(int(len), self.max_desc_len)

            rand_offset=random.randint(0, self.data_len-1)
            len, pos = self.idx_descs[rand_offset]['length'], self.idx_descs[rand_offset]['pos']
            bad_desc = self.descs[pos:pos+len].astype('int64')
            bad_desc = self.pad_seq(bad_desc, self.max_desc_len)
            bad_desc_len=min(int(len), self.max_desc_len)

            return name, name_len, apiseq, api_len, tokens, tok_len, good_desc, good_desc_len, bad_desc, bad_desc_len
        return name, name_len, apiseq, api_len, tokens, tok_len
        
    def __len__(self):
        return self.data_len
    

def load_dict(filename):
    return json.loads(open(filename, "r").readline())
    #return pickle.load(open(filename, 'rb')) 

def load_vecs(fin):         
    """read vectors (2D numpy array) from a hdf5 file"""
    h5f = tables.open_file(fin)
    h5vecs= h5f.root.vecs
    
    vecs=np.zeros(shape=h5vecs.shape,dtype=h5vecs.dtype)
    vecs[:]=h5vecs[:]
    h5f.close()
    return vecs
        
def save_vecs(vecs, fout):
    fvec = tables.open_file(fout, 'w')
    atom = tables.Atom.from_dtype(vecs.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = fvec.create_carray(fvec.root,'vecs', atom, vecs.shape,filters=filters)
    ds[:] = vecs
    print('done')
    fvec.close()

if __name__ == '__main__':
    
    input_dir='./data/github/'
    VALID_FILE=input_dir+'train.h5'
    valid_set=CodeSearchDataset(VALID_FILE)
    valid_data_loader=torch.utils.data.DataLoader(dataset=valid_set,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1)
    vocab = load_dict(input_dir+'vocab.json')
    ivocab = {v: k for k, v in vocab.items()}
    #print ivocab
    k=0
    for qapair in valid_data_loader:
        k+=1
        if k>20:
            break
        decoded_words=[]
        idx=qapair[0].numpy().tolist()[0]
        print (idx)
        for i in idx:
            decoded_words.append(ivocab[i])
        question = ' '.join(decoded_words)
        decoded_words=[]
        idx=qapair[1].numpy().tolist()[0]
        for i in idx:
            decoded_words.append(ivocab[i])
        answer=' '.join(decoded_words)
        print('<', question)
        print('>', answer)
