
# Experiments on CIFAR-10
## Usage

1. In order to apply ADMM-based sparse algorithm execute:

 python ADMM_cifar.py --model_id=0 --sparsity_function_id=0 --data_path=path-to-data --ckpt_path_pretrained=loading-path-to-pretrained-net --ckpt_path_ADMM=saving-path-for-results 
 * The function gets the path for loading pretrained weights and also the path for saving ADMM ckpt output results. If the result path contains previously saved ckpt files, it loads latest of them and continues training. otherwise it loads the pretrained weights and starts training.

2. In order to validate the sparce network execute:

 python validate_cifar.py --model_id=0(nin)/1(nin_c3)/2(nin_c3_lr) --data_path=path-to-data --ckpt_path=path-to-results --task=0(pretrained model)/1(ADMM sparse CNN)
