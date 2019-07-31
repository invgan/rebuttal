import os
from multiprocessing import Pool
import yaml
import argparse

"""
Using param_sweep
Specify the parameter configs to sweep
Number of GPUs can be specified. The code runs parameter configs across the GPUs in parallel
"""

# Param sweep for reconstruct

def run_process(config):
    print('Running gpu {}, encoder loss type {}, encoder_lr {} ...'.format(config[0], config[1], config[2]))
    cmd = 'CUDA_VISIBLE_DEVICES={} python train.py --train_encoder --test_encoder --train_iters 10000 --debug --encoder_loss_type {} --encoder_lr {} --cfg {} --init_path {}'.format(config[0], config[1], config[2], config[3], config[4])
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, help='path to param sweep config file')
    args = parser.parse_args()

    with open(args.cfg) as f:
        sweep_params = yaml.load(f)
        encoder_loss_type = sweep_params['LOSS_TYPE']
        encoder_lr_list = sweep_params['LR']
        config_file = sweep_params['CFG']
        init_file = sweep_params['INIT_PATH']
        num_gpus = sweep_params['NUM_GPUS']
    
    run_configs = []
    counter = 0
    
    for encoder_lr in encoder_lr_list:
        run_config = (counter%num_gpus, encoder_loss_type, encoder_lr, config_file, init_file)
        run_configs.append(run_config)
        counter += 1

    pool = Pool(processes=num_gpus)
    pool.map(run_process, run_configs, chunksize=1)


if __name__ == '__main__':
    main()
            
