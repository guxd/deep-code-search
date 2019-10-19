
def config_JointEmbeder():   
    conf = {
        'data_path':'./data/github/', # location of the data corpus
        # data_params
        'dataset_name':'CodeSearchDataset', # name of dataset to specify a data loader
            #training data
            'train_name':'train.name.h5',
            'train_api':'train.apiseq.h5',
            'train_tokens':'train.tokens.h5',
            'train_desc':'train.desc.h5',
            #test data
            'valid_name':'valid.name.h5',
            'valid_api':'valid.apiseq.h5',
            'valid_tokens':'valid.tokens.h5',
            'valid_desc':'valid.desc.h5',
            #use data (computing code vectors)
            'use_codebase':'use.rawcode.txt',#'use.rawcode.h5'
            'use_names':'use.name.h5',
            'use_apis':'use.apiseq.h5',
            'use_tokens':'use.tokens.h5',     
            #results data(code vectors)            
            'use_codevecs':'use.codevecs.normalized.h5',#'use.codevecs.h5',         
                   
            #parameters
            'name_len': 6,
            'api_len':30,
            'tokens_len':50,
            'desc_len': 30,
            'n_words': 10000, # len(vocabulary) + 1
            #vocabulary info
            'vocab_name':'vocab.name.json',
            'vocab_api':'vocab.apiseq.json',
            'vocab_tokens':'vocab.tokens.json',
            'vocab_desc':'vocab.desc.json',
                    
        #training_params            
            'batch_size': 64,
            'chunk_size':100000,
            'nb_epoch': 5,
            'validation_split': 0.2,
            #'optimizer': 'adam',
            'lr':1e-3,
            'valid_every': 1000,
            'n_eval': 100,
            'evaluate_all_threshold': {
                'mode': 'all',
                'top1': 0.4,
            },
            'log_every': 100,
            'save_every': 5000,
            'reload':-1,#970,#epoch that the model is reloaded from . If reload=0, then train from scratch
        

        # model_params
            'emb_size': 100,
            'n_hidden': 1000,#number of hidden dimension of code/desc representation
            # recurrent
            'lstm_dims': 500, # * 2
            'init_embed_weights_methname': None,#'word2vec_100_methname.h5', 
            'init_embed_weights_tokens': None,#'word2vec_100_tokens.h5', 
            'init_embed_weights_desc': None,#'word2vec_100_desc.h5',           
            'margin': 0.05,
            'sim_measure':'cos',#similarity measure: gesd, cosine, aesd
         
    }
    return conf








