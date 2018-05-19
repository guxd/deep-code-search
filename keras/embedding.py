from __future__ import print_function

import os
import sys
import random
import re
import pickle
import numpy

from gensim.models import Word2Vec

from models import *

random.seed(42)


def load(path, name):
    return pickle.load(open(os.path.join(path, name), 'rb'))


def revert(vocab, indices):
    return [vocab.get(i, 'X') for i in indices]

def sample_weights_classic(sizeX, sizeY, sparsity, scale, rng):
    sizeX = int(sizeX)
    sizeY = int(sizeY)
    if sparsity < 0:
        sparsity = sizeY
    else:
        sparsity = numpy.minimum(sizeY, sparsity)
    sparsity = numpy.minimum(sizeY, sparsity)
    values = numpy.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals
    return values.astype(theano.config.floatX)


if __name__ == '__main__':
    workdir='D:/workspace/Code2Doc/resources/data/github/codesearch/'
    #vocab_file=workdir+'vocab.methname.pkl'
    #vocab_file=workdir+'vocab.tokens.pkl'
    vocab_file=workdir+'vocab.desc.pkl'
    train_file=workdir+'soquestions.txt'
    #emb_file=workdir+'word2vec_100_methname.h5'
    #emb_file=workdir+'word2vec_100_tokens.h5'
    emb_file=workdir+'word2vec_100_desc.h5'
    vocab_size=10000
    emb_size=100
        

    sentences = list()
    train_data = open(train_file,'r').readlines()
    random.shuffle(train_data)
    for txt in train_data:
        s = re.split('\W+',txt)
        s = [token.lower() for token in s if token != '' and token != '\n']
        sentences.append(s)

    model = Word2Vec(sentences, size=emb_size, min_count=5, window=5, sg=1, workers=30, iter=50)
    weights = model.syn0
    d = dict([(k, v.index) for k, v in model.vocab.items()])

    # this is the stored weights of an equivalent embedding layer
    #emb = np.load('models/embedding_100_dim.h5')
    #emb = np.zeros((vocab_size,100))
    rng = numpy.random.RandomState(1234)
    emb = sample_weights_classic(vocab_size, 100, -1, 0.01, rng)

    # load the vocabulary
    vocab = pickle.load(open(vocab_file, 'rb'))

    # swap the word2vec weights with the embedded weights
    for i, w in vocab.items():
        if w not in d: continue
        emb[i, :] = weights[d[w], :]

    np.save(open(emb_file, 'wb'), emb)
