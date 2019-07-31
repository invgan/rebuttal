import os
from multiprocessing import Pool

"""
Using param_sweep
Specify the parameter configs to sweep
Number of GPUs can be specified. The code runs parameter configs across the GPUs in parallel
"""

# Param sweep for reconstruct

def run_process(config):
    print('Running gpu %d, rec-iter %d, rec-rr %d, rec_lr %f ...'%(config[0], config[1], config[2], config[3]))
    cmd = 'CUDA_VISIBLE_DEVICES=%d python train.py --save_recs --cfg experiments/cfgs/gans/cifar10_hinge_resnet.yml --debug --init_path output/gans/cifar10_hinge_resnet --rec_iters %d --rec_rr %d --rec_lr %f'%(config[0], config[1], config[2], config[3])
    os.system(cmd)

# Params to sweep
rec_iter_list = [1, 2, 3]
rec_rr_list = [1, 4, 8]
rec_lr_list = [10, 1, 0.1]

configs = []
num_gpus = 2
counter = 0

for rec_iter in rec_iter_list:
    for rec_rr in rec_rr_list:
        for rec_lr in rec_lr_list:
            cfg = (counter%num_gpus, rec_iter, rec_rr, rec_lr)
            configs.append(cfg)
            counter += 1

pool = Pool(processes=num_gpus)
pool.map(run_process, configs, chunksize=1)
            
"""

# param sweep for encoder

def run_process(config):
    print('Running gpu %d, encoder loss type %s, encoder_lr %f ...'%(config[0], config[1], config[2]))
    cmd = 'CUDA_VISIBLE_DEVICES=%d python train.py --train_encoder --test_encoder --cfg experiments/cfgs/gans/cifar10_hinge_resnet.yml --train_iters 10000 --debug --encoder_loss_type %s --encoder_lr %f --init_path output/gans/cifar10_hinge_resnet'%(config[0], config[1], config[2])
    os.system(cmd)


## Params to sweep       
encoder_loss_type_list = ['margin']
encoder_lr_list = [0.0001, 0.0002, 0.0004]

configs = []
num_gpus = 2
counter = 0

for encoder_loss_type in encoder_loss_type_list:
    for encoder_lr in encoder_lr_list:
        cfg = (counter%num_gpus, encoder_loss_type, encoder_lr)
        configs.append(cfg)
        counter += 1

pool = Pool(processes=num_gpus)
pool.map(run_process, configs, chunksize=1)
"""
