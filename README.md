# sparsecnn
Implementation of the DL-ELM network architecture
This repository contains an implementation of the sparse convolutional neural network using alternating direction method of multiplier (ADMM) introduced by F. Kiaee, C. Gagné, and M. Abbasi. The original paper can be found at
[http://dx.doi.org/10.1016/j.neucom.2016.08.011](http://dx.doi.org/10.1016/j.neucom.2016.08.011)

## Contents

1. Python implementation of ADMM-based sparse CNN (ADMM.py)
2. A module that constructs the network in network (NIN) graph in tensorflow and loads the pretrained weights (ADMMmodels.py)
3. (DL_ELM_demo.m)

For the record, it gets about 80% accuracy on the 
test set. 3. Spect benchmark dataset for demonstration (SPECT.train.txt, SPECT.test.txt)
