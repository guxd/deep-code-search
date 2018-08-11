# Deep Code Search

  Code for the ICSE 2018 paper [Deep Code Search](https://guxd.github.io/papers/deepcs.pdf).

## Two Versions
We release both ```Keras``` and ```PyTorch``` code of our approach, in the ```keras``` and ```pytorch``` folders, respectively.

- The ```Keras``` folder contains the code to run the experiments presented in the paper. The code is frozen to what it was when we originally wrote the paper. (NOTE: we modified some deprecated API invocations to fit for the latest Keras and theano).

- The ```PyTorch``` is the bleeding-edge reporitory where we packaged it up, improved the code quality and added some features.

If you are interested in using DeepCS, check out the PyTorch version and feel free to contribute.

For more information, please refer to the README files under the directory of each component.



## Tool Demo

An online tool demo can be found in http://211.249.63.55:81/

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
