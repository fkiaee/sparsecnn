# sparsecnn
Implementation of ADMM-based sparse CNN architecture.
This repository contains an implementation of the sparse convolutional neural network using alternating direction method of multiplier (ADMM) introduced by F. Kiaee, C. Gagné, and M. Abbasi. The original paper can be found at
[http://dx.doi.org/10.1016/j.neucom.2016.08.011](http://dx.doi.org/10.1016/j.neucom.2016.08.011)

## Contents

1. Python implementation of ADMM-based sparse CNN (ADMM.py)
2. A module that constructs the network in network (NIN) graph in tensorflow and loads the pretrained weights (ADMMmodels.py)
3. Utility functions for use with the ADMM.py. There are functions to load Cifar10 data and to apply l0 or l1 sparsity promoting penalty functions(ADMMutils.py)
4. Pretrain of the network and saving .ckpt weights (.py) 
