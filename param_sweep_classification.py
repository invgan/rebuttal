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
    print('Running gpu {}, rec iters {}, rec lr {} ...'.format(config[0], config[1], config[2]))
    cmd = 'CUDA_VISIBLE_DEVICES={} python classification.py --rec_iters {} --rec_lr {} --cfg {}'.format(
        config[0], config[1], config[2], config[3])
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, help='path to param sweep config file')
    args = parser.parse_args()

    with open(args.cfg) as f:
        sweep_params = yaml.load(f)
        rec_iter_list = sweep_params['REC_ITER']
        rec_lr_list = sweep_params['REC_LR']
        config_file = sweep_params['CFG']
        num_gpus = sweep_params['NUM_GPUS']

    run_configs = []
    counter = 0

    for rec_iter in rec_iter_list:
        for rec_lr in rec_lr_list:
            run_config = (counter%num_gpus, rec_iter, rec_lr, config_file)
            run_configs.append(run_config)
            counter += 1

    pool = Pool(processes=num_gpus)
    pool.map(run_process, run_configs, chunksize=1)


if __name__ == '__main__':
    main()