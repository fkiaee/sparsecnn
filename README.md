# sparsecnn
Implementation of ADMM-based sparse CNN architecture.
This repository contains an implementation of the sparse convolutional neural network using alternating direction method of multiplier (ADMM) introduced by F. Kiaee, C. Gagné, and M. Abbasi. The original paper can be found at
[https://arxiv.org/abs/1611.01590](https://arxiv.org/abs/1611.01590)

## Contents

1. Python implementation of ADMM-based sparse CNN (ADMM.py)
2. A module that constructs the network in network (NIN) graph in tensorflow and loads the pretrained weights (ADMMmodels.py)
3. Utility functions for use with the ADMM.py. There are functions to load Cifar10 data and to apply norm-0 or norm-1 sparsity promoting penalty functions(ADMMutils.py)
4. Pretraining of the network and saving .ckpt weights (pretraining.py) 
5. Validation of the pretraind network or ADMM-based sparse network (validate.py)

## Usage

Make sure you're using the tensorflow 0.9.0 version.

1. Run pretrain.py to provide a pretrained initial weights for ADMM-based sparse algorithm.
2. In order to validate the pretrain network (ACC = 82.3 %) execute:
 python validate.py --model_id=0 --task=0 
3. Run ADMM.py to apply ADMM-based sparse algorithm
4. In order to validate the sparce network execute:
 python validate.py --model_id=0 --task=1
