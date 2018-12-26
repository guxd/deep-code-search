from __future__ import print_function
from __future__ import absolute_import
import os
from keras.engine import Input
from keras.layers import Concatenate, Dot, Embedding, Dropout, Lambda, Activation, LSTM, Dense
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
import numpy as np
import logging
logger = logging.getLogger(__name__)
    
class JointEmbeddingModel:
    def __init__(self, config):
        self.config = config
        self.model_params = config.get('model_params', dict())
        self.data_params = config.get('data_params',dict())
        self.methname = Input(shape=(self.data_params['methname_len'],), dtype='int32', name='i_methname')
        self.apiseq= Input(shape=(self.data_params['apiseq_len'],),dtype='int32',name='i_apiseq')
        self.tokens=Input(shape=(self.data_params['tokens_len'],),dtype='int32',name='i_tokens')
        self.desc_good = Input(shape=(self.data_params['desc_len'],), dtype='int32', name='i_desc_good')
        self.desc_bad = Input(shape=(self.data_params['desc_len'],), dtype='int32', name='i_desc_bad')
        
        # initialize a bunch of variables that will be set later
        self._code_repr_model=None
        self._desc_repr_model=None        
        self._sim_model = None        
        self._training_model = None
        #self.prediction_model = None
        
        #create a model path to store model info
        if not os.path.exists(self.config['workdir']+'models/'+self.model_params['model_name']+'/'):
            os.makedirs(self.config['workdir']+'models/'+self.model_params['model_name']+'/')
    
    def build(self):
        '''
        1. Build Code Representation Model
        '''
        logger.debug('Building Code Representation Model')
        methname = Input(shape=(self.data_params['methname_len'],), dtype='int32', name='methname')
        apiseq= Input(shape=(self.data_params['apiseq_len'],),dtype='int32',name='apiseq')
        tokens=Input(shape=(self.data_params['tokens_len'],),dtype='int32',name='tokens')
        
        ## method name representation ##
        #1.embedding
        init_emb_weights = np.load(self.config['workdir']+self.model_params['init_embed_weights_methname']) if self.model_params['init_embed_weights_methname'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        embedding = Embedding(input_dim=self.data_params['n_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              weights=init_emb_weights,
                              mask_zero=False,#Whether 0 in the input is a special "padding" value that should be masked out. 
                              #If set True, all subsequent layers in the model must support masking, otherwise an exception will be raised.
                              name='embedding_methname')
        methname_embedding = embedding(methname)
        dropout = Dropout(0.25,name='dropout_methname_embed')
        methname_dropout = dropout(methname_embedding)
        #2.rnn
        f_rnn = LSTM(self.model_params.get('n_lstm_dims', 128), recurrent_dropout=0.2, 
                     return_sequences=True, name='lstm_methname_f')
        
        b_rnn = LSTM(self.model_params.get('n_lstm_dims', 128), return_sequences=True, 
                     recurrent_dropout=0.2, name='lstm_methname_b',go_backwards=True)        
        methname_f_rnn = f_rnn(methname_dropout)
        methname_b_rnn = b_rnn(methname_dropout)
        dropout = Dropout(0.25,name='dropout_methname_rnn')
        methname_f_dropout = dropout(methname_f_rnn)
        methname_b_dropout = dropout(methname_b_rnn)
        #3.maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),name='maxpool_methname')
        methname_pool = Concatenate(name='concat_methname_lstms')([maxpool(methname_f_dropout), maxpool(methname_b_dropout)])
        activation = Activation('tanh',name='active_methname')
        methname_repr = activation(methname_pool)
        
        
        ## API Sequence Representation ##
        #1.embedding
        embedding = Embedding(input_dim=self.data_params['n_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              #weights=weights,
                              mask_zero=False,#Whether 0 in the input is a special "padding" value that should be masked out. 
                                         #If set True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_apiseq')
        apiseq_embedding = embedding(apiseq)
        dropout = Dropout(0.25,name='dropout_apiseq_embed')
        apiseq_dropout = dropout(apiseq_embedding)
        #2.rnn
        f_rnn = LSTM(self.model_params.get('n_lstm_dims', 100), return_sequences=True, recurrent_dropout=0.2,
                      name='lstm_apiseq_f')
        b_rnn = LSTM(self.model_params.get('n_lstm_dims', 100), return_sequences=True, recurrent_dropout=0.2, 
                      name='lstm_apiseq_b', go_backwards=True)        
        apiseq_f_rnn = f_rnn(apiseq_dropout)
        apiseq_b_rnn = b_rnn(apiseq_dropout)
        dropout = Dropout(0.25,name='dropout_apiseq_rnn')
        apiseq_f_dropout = dropout(apiseq_f_rnn)
        apiseq_b_dropout = dropout(apiseq_b_rnn)
        #3.maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),name='maxpool_apiseq')
        apiseq_pool = Concatenate(name='concat_apiseq_lstms')([maxpool(apiseq_f_dropout), maxpool(apiseq_b_dropout)])
        activation = Activation('tanh',name='active_apiseq')
        apiseq_repr = activation(apiseq_pool)
        
        
        ## Tokens Representation ##
        #1.embedding
        init_emb_weights = np.load(self.config['workdir']+self.model_params['init_embed_weights_tokens']) if self.model_params['init_embed_weights_tokens'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        embedding = Embedding(input_dim=self.data_params['n_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              weights=init_emb_weights,
                              #mask_zero=True,#Whether 0 in the input is a special "padding" value that should be masked out. 
                              #If set True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_tokens')
        tokens_embedding = embedding(tokens)
        dropout = Dropout(0.25,name='dropout_tokens_embed')
        tokens_dropout= dropout(tokens_embedding)

        #4.maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),name='maxpool_tokens')
        tokens_pool = maxpool(tokens_dropout)
        activation = Activation('tanh',name='active_tokens')
        tokens_repr= activation(tokens_pool)        
        
        ## concatenate the representation of code ##
        merged_methname_api=Concatenate(name='merge_methname_api')([methname_repr,apiseq_repr])
        merged_code_repr=Concatenate(name='merge_coderepr')([merged_methname_api,tokens_repr])
        code_repr=Dense(self.model_params.get('n_hidden',400),activation='tanh',name='dense_coderepr')(merged_code_repr)
        
        
        self._code_repr_model=Model(inputs=[methname,apiseq,tokens],outputs=[code_repr],name='code_repr_model')
        print('\nsummary of code representation model')
        self._code_repr_model.summary()
        fname=self.config['workdir']+'models/'+self.model_params['model_name']+'/_code_repr_model.png'
        #plot_model(self._code_repr_model, show_shapes=True, to_file=fname)
        
        
        
        
        
        '''
        2. Build Desc Representation Model
        '''
        ## Desc Representation ##
        logger.debug('Building Desc Representation Model')
        desc = Input(shape=(self.data_params['desc_len'],), dtype='int32', name='desc')
        #1.embedding
        init_emb_weights = np.load(self.config['workdir']+self.model_params['init_embed_weights_desc']) if self.model_params['init_embed_weights_desc'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        embedding = Embedding(input_dim=self.data_params['n_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              weights=init_emb_weights,
                              mask_zero=True,#Whether 0 in the input is a special "padding" value that should be masked out. 
                                      #If set True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_desc')
        desc_embedding = embedding(desc)
        dropout = Dropout(0.25,name='dropout_desc_embed')
        desc_dropout = dropout(desc_embedding)
        #2. rnn
        f_rnn = LSTM(self.model_params.get('n_lstm_dims', 100), return_sequences=True, recurrent_dropout=0.2,
                     name='lstm_desc_f')
        b_rnn = LSTM(self.model_params.get('n_lstm_dims', 100), return_sequences=True, recurrent_dropout=0.2, 
                     name='lstm_desc_b', go_backwards=True) 
        desc_f_rnn = f_rnn(desc_dropout)
        desc_b_rnn = b_rnn(desc_dropout)
        dropout = Dropout(0.25,name='dropout_desc_rnn')
        desc_f_dropout = dropout(desc_f_rnn)
        desc_b_dropout = dropout(desc_b_rnn)
        #3. maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),name='maxpool_desc')
        desc_pool = Concatenate(name='concat_desc_rnns')([maxpool(desc_f_dropout), maxpool(desc_b_dropout)])
        activation = Activation('tanh',name='active_desc')
        desc_repr = activation(desc_pool)
        
        self._desc_repr_model=Model(inputs=[desc],outputs=[desc_repr],name='desc_repr_model')
        print('\nsummary of desc representation model')
        self._desc_repr_model.summary()
        fname=self.config['workdir']+'models/'+self.model_params['model_name']+'/_desc_repr_model.png'
        #plot_model(self._desc_repr_model, show_shapes=True, to_file=fname)               
        
            
        """
        3: calculate the cosine similarity between code and desc
        """     
        logger.debug('Building similarity model') 
        code_repr=self._code_repr_model([methname,apiseq,tokens])
        desc_repr=self._desc_repr_model([desc])
        cos_sim=Dot(axes=1, normalize=True, name='cos_sim')([code_repr, desc_repr])
        
        sim_model = Model(inputs=[methname,apiseq,tokens,desc], outputs=[cos_sim],name='sim_model')   
        self._sim_model=sim_model  #for model evaluation  
        print ("\nsummary of similarity model")
        self._sim_model.summary() 
        fname=self.config['workdir']+'models/'+self.model_params['model_name']+'/_sim_model.png'
        #plot_model(self._sim_model, show_shapes=True, to_file=fname)
        
        
        '''
        4:Build training model
        '''
        good_sim = sim_model([self.methname,self.apiseq,self.tokens, self.desc_good])# similarity of good output
        bad_sim = sim_model([self.methname,self.apiseq,self.tokens, self.desc_bad])#similarity of bad output
        loss = Lambda(lambda x: K.maximum(1e-6, self.model_params['margin'] - x[0] + x[1]),
                     output_shape=lambda x: x[0], name='loss')([good_sim, bad_sim])

        logger.debug('Building training model')
        self._training_model=Model(inputs=[self.methname,self.apiseq,self.tokens,self.desc_good,self.desc_bad],
                                   outputs=[loss],name='training_model')
        print ('\nsummary of training model')
        self._training_model.summary()      
        fname=self.config['workdir']+'models/'+self.model_params['model_name']+'/_training_model.png'
        #plot_model(self._training_model, show_shapes=True, to_file=fname)     

    def compile(self, optimizer, **kwargs):
        logger.info('compiling models')
        self._code_repr_model.compile(loss='cosine_proximity', optimizer=optimizer, **kwargs)
        self._desc_repr_model.compile(loss='cosine_proximity', optimizer=optimizer, **kwargs)
        self._training_model.compile(loss=lambda y_true, y_pred: y_pred+y_true-y_true, optimizer=optimizer, **kwargs)
        #+y_true-y_true is for avoiding an unused input warning, it can be simply +y_true since y_true is always 0 in the training set.
        self._sim_model.compile(loss='binary_crossentropy', optimizer=optimizer, **kwargs)

    def fit(self, x, **kwargs):
        assert self._training_model is not None, 'Must compile the model before fitting data'
        y = np.zeros(shape=x[0].shape[:1],dtype=np.float32)
        return self._training_model.fit(x, y, **kwargs)

    def repr_code(self, x, **kwargs):
        return self._code_repr_model.predict(x, **kwargs)
    
    def repr_desc(self, x, **kwargs):
        return self._desc_repr_model.predict(x, **kwargs)
    
    def predict(self, x, **kwargs):
        return self._sim_model.predict(x, **kwargs)

    def save(self, code_model_file, desc_model_file, **kwargs):
        assert self._code_repr_model is not None, 'Must compile the model before saving weights'
        self._code_repr_model.save_weights(code_model_file, **kwargs)
        assert self._desc_repr_model is not None, 'Must compile the model before saving weights'
        self._desc_repr_model.save_weights(desc_model_file, **kwargs)

    def load(self, code_model_file, desc_model_file, **kwargs):
        assert self._code_repr_model is not None, 'Must compile the model loading weights'
        self._code_repr_model.load_weights(code_model_file, **kwargs)
        assert self._desc_repr_model is not None, 'Must compile the model loading weights'
        self._desc_repr_model.load_weights(desc_model_file, **kwargs)

 
 
 
 