# Deep Code Search

PyTorch implementation of [Deep Code Search](https://guxd.github.io/papers/deepcs.pdf).

## Dependency
> Tested in MacOS 10.12, Ubuntu 16.04
* Python 3.6
* PyTorch 
* tqdm

 ```
 pip install -r requirements.txt
 ```
 

## Code Structures

 - `models`: neural network models for code/desc representation and similarity measure.
 - `modules.py`: basic modules for model construction.
 - `train.py`: train and validate code/desc representaton models; 
 - `repr_code.py`: encode code into vectors and store them to a file; 
 - `search.py`: perform code search;
 - `configs.py`: configurations for models defined in the `models` folder. 
   Each function defines the hyper-parameters for the corresponding model.
 - `data_loader.py`: A PyTorch dataset loader.
 - `utils.py`: utilities for models and training. 

## Pretrained Model
   If you want a quick test, [here]() is a pretrained model. Put it in `./output/JointEmbeder/github/models/` and run:
   ```
   python search.py --reload_from 
   ```
 
## Usage

   ### Data Preparation
  The `/data` folder provides a small dummy dataset for quick deployment.  
  To train and test our model:
  
  1) Download and unzip real dataset from [Google Drive](https://drive.google.com/drive/folders/1GZYLT_lzhlVczXjD6dgwVUvDDPHMB6L7?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1U_MtFXqq0C-Qh8WUFAWGvg) for Chinese users.
  
  2) Replace each file in the `/data` folder with the corresponding real file. 
  
   ### Configuration
   Edit hyper-parameters and settings in `config.py`

   ### Train
   
   ```bash
   python train.py --model JointEmbeder
   ```
   
   ### Code Embedding
   
   ```bash
   python repr_code.py --model JointEmbeder --reload_from XXX
   ```
   
   ### Search
   
   ```bash
   python search.py --model JointEmbeder --_reload_from XXX
   ```
   

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
