# Deep Code Search
A keras implementation of the paper [Deep Code Search](https://guxd.github.io/papers/deepcs.pdf).

## Dependency
> Tested in Ubuntu 16.04
* Python 2.7-3.6
* Keras 2.0.0 or newer
* Tensorflow or Theano 0.8.0~0.9.1

## Code Structures

 - `models.py`: Neural network models for code/desc representation and similarity measure.
 
 - `codesearcher.py`: The main entry for code search, including four sub-tasks: 
     1) Train: train code/desc representaton models; 
     2) Eval: evaluate the learnt code/desc representation models; 
     3) Code Embedding: encode code into vectors and store them to a file; 
     4) Search: search relevant code for a given query.
     
 - `configs.py`: Configurations for models defined in the `models.py`. 
   Each function defines the hyperparameters for the corresponding model.


## Usage

   ### Data Preparation
  The `/data` folder provides a small sample dataset for quick running. 
  To train and test our model:
  
  1) Download and unzip real dataset from [Google Drive](https://drive.google.com/drive/folders/1GZYLT_lzhlVczXjD6dgwVUvDDPHMB6L7?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1U_MtFXqq0C-Qh8WUFAWGvg) for Chinese users.
  
  2) Replace each file in the `/data` folder with the corresponding real file. 
  
   ### Configuration
   
   Edit hyper-parameters and settings in `config.py`
   
   ### Train
   
   ```bash
   python codesearcher.py --mode train
   ```
   
   ### Code Embedding
   
   ```bash
   python codesearcher.py --mode repr_code
   python norm_code_repr.py
   ```
   
   ### Search
   
   ```bash
   python codesearcher.py --mode search
   ``` 
   
## Tool Demo

An online tool demo can be found at http://211.249.63.55:81/ 

## Citation
If you find it useful and would like to cite it, the following would be appropriate:
```
@inproceedings{gu2018deepcs,
  title={Deep Code Search},
  author={Gu, Xiaodong and Zhang, Hongyu and Kim, Sunghun},
  booktitle={Proceedings of the 2018 40th International Conference on Software Engineering (ICSE 2018)},
  year={2018},
  organization={ACM}
}
```
