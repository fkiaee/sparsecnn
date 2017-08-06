# sparsecnn
Implementation of ADMM-based sparse CNN architecture.
This repository contains an implementation of the sparse convolutional neural network using alternating direction method of multiplier (ADMM) introduced by F. Kiaee, C. Gagné, and M. Abbasi. The original paper can be found at
[https://arxiv.org/abs/1611.01590](https://arxiv.org/abs/1611.01590)

## Contents

1. A module that constructs the graph of investiated models in tensorflow  (cifar_models.py and imagenet_models.py)
2. Python implementation of ADMM-based sparse CNN (ADMM_cifar.py and ADMM_imagenet.py)
3. Utility functions for use with the ADMM algorithm. There are functions to load Cifar10 and imagenet data and to apply norm-0 or norm-1 sparsity promoting penalty functions(ADMMutils.py)
4. Pretrained weights in ckpt format (pretrained folder) 
4. Validation of the pretraind network or ADMM-based sparse network (validate_cifar.py and validate_imagenet.py)


## Implementation Hints
1. The name of the variables is consistent with the paper.
    * The filters subject to sparse structural constraints are named *W_c*/*W_f* for convolutional/fully_connected layers and their    corresponding biases are named *b_c*/*b_f*. 
    * *Gamma* is the dual variable (i.e., the Lagrange multiplier) and *F* is an additional variable to introduce an additional constraint *W - F = 0* giving rise to decoupling the objective function. 
    * All the mentioned variable are added to *variable_dict* dictionary. 
2. Dual variable, *Gamma*, and sparsity promoting variable, *F*, are updated manually and are not trained (trainable = False). These two set of variables are then included in the 'non_trainable_variables' collection to ask the saver to save their final updated values.
3. Update equations (9) and (10) corresponding to the norm-1 or norm-0 sparsity promoting steps are implemented in *block_shrinkage* and *block_truncate* functions, respectively. The sparse filter tensors and index of zero filters are passed to *F_new* and *zero_ind* operations, respectively.
4. Stopping criterion for ADMM is investigated using non-trainable variables resF=||F_new-F|| and resWF=||F_new-W||. 
5. Variable *F* is then updated with its updated value *F_new* using *assign_F* operations.
6. Similarly, *assign_Gammas* operations are used to implement the updating equation (6). 
7. Placeholders *zero_map* contains tensors of all-zero/all-one filter patterns which are built based on the *zero_ind* operations. The placeholders are used to fix sparsity patterns of weights during finetuning step.
