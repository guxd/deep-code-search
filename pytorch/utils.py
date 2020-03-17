import numpy as np
import time
import math
import torch
from torch.nn import functional as F

PAD_ID, SOS_ID, EOS_ID, UNK_ID = [0, 1, 2, 3]

def cos_approx(data1,data2):
    """numpy implementation of cosine similarity for matrix"""
    #print("warning: the second matrix will be transposed, so try to put the simpler matrix as the second argument in order to save time.")
    dotted = np.dot(data1,np.transpose(data2))
    norm1 = np.linalg.norm(data1,axis=1)
    norm2 = np.linalg.norm(data2,axis=1)
    matrix_vector_norms = np.multiply(norm1, norm2)
    neighbors = np.divide(dotted, matrix_vector_norms)
    return neighbors

def normalize(data):
    """normalize matrix by rows"""
    return data/np.linalg.norm(data,axis=1,keepdims=True)

def dot_np(data1,data2):
    """cosine similarity for normalized vectors"""
    #print("warning: the second matrix will be transposed, so try to put the simpler matrix as the second argument in order to save time.")
    return np.dot(data1, data2.T)

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 

def similarity(vec1, vec2, measure='cos'):
    if measure=='cos':
        vec1_norm = normalize(vec1)
        vec2_norm = normalize(vec2)
        return np.dot(vec1_norm, vec2_norm.T)[:,0]
    elif measure=='poly':
        return (0.5*np.dot(vec1, vec2.T).diagonal()+1)**2
    elif measure=='sigmoid':
        return np.tanh(np.dot(vec1, vec2.T).diagonal()+1)
    elif measure in ['euc', 'gesd', 'aesd']: #https://arxiv.org/pdf/1508.01585.pdf 
        euc_dist = np.linalg.norm(vec1-vec2, axis=1)
        euc_sim = 1 / (1 + euc_dist)
        if measure=='euc': return euc_sim                
        sigmoid_sim = sigmoid(np.dot(vec1, vec2.T).diagonal()+1)
        if measure == 'gesd': return euc_sim * sigmoid_sim
        elif measure == 'aesd': return 0.5*(euc_sim+sigmoid_sim)

#######################################################################

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%d:%d'% (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s<%s'%(asMinutes(s), asMinutes(rs))

#######################################################################
import nltk
try: nltk.word_tokenize("hello world")
except LookupError: nltk.download('punkt')
    
def sent2indexes(sentence, vocab, maxlen):
    '''sentence: a string or list of string
       return: a numpy array of word indices
    '''      
    def convert_sent(sent, vocab, maxlen):
        idxes = np.zeros(maxlen, dtype=np.int64)
        idxes.fill(PAD_ID)
        tokens = nltk.word_tokenize(sent.strip())
        idx_len = min(len(tokens), maxlen)
        for i in range(idx_len): idxes[i] = vocab.get(tokens[i], UNK_ID)
        return idxes, idx_len
    if type(sentence) is list:
        inds, lens = [], []
        for sent in sentence:
            idxes, idx_len = convert_sent(sent, vocab, maxlen)
            #idxes, idx_len = np.expand_dims(idxes, 0), np.array([idx_len])
            inds.append(idxes)
            lens.append(idx_len)
        return np.vstack(inds), np.vstack(lens)
    else:
        inds, lens = sent2indexes([sentence], vocab, maxlen)
        return inds[0], lens[0]
    
def indexes2sent(indexes, vocab, ignore_tok=PAD_ID): 
    '''indexes: numpy array'''
    def revert_sent(indexes, ivocab, ignore_tok=PAD_ID):
        indexes=filter(lambda i: i!=ignore_tok, indexes)
        toks, length = [], 0        
        for idx in indexes:
            toks.append(ivocab.get(idx, '<unk>'))
            length+=1
            if idx == EOS_ID:
                break
        return ' '.join(toks), length
    
    ivocab = {v: k for k, v in vocab.items()}
    if indexes.ndim==1:# one sentence
        return revert_sent(indexes, ivocab, ignore_tok)
    else:# dim>1
        sentences, lens =[], [] # a batch of sentences
        for inds in indexes:
            sentence, length = revert_sent(inds, ivocab, ignore_tok)
            sentences.append(sentence)
            lens.append(length)
        return sentences, lens

########################################################################
