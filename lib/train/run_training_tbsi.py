import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import sys
import argparse
import importlib
import cv2 as cv
import torch.backends.cudnn
import torch.distributed as dist

import random
import numpy as np
torch.backends.cudnn.benchmark = False

import _init_paths
import lib.train.admin.settings as ws_settings


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(seed)
    print("Random seeds are initialized.")


def run_training(script_name, config_name, cudnn_benchmark=True, local_rank=-1, save_dir=None, base_seed=None,
                 use_lmdb=False, script_name_prv=None, config_name_prv=None, use_wandb=False,
                 distill=None, script_teacher=None, config_teacher=None):
    """Run the train script.
    args:
        script_name: Name of emperiment in the "experiments/" folder.
        config_name: Name of the yaml file in the "experiments/<script_name>".
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """
    if save_dir is None:
        print("save_dir dir is not given. Use the default dir instead.")
    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    print('script_name: {}.py  config_name: {}.yaml'.format(script_name, config_name))

    '''2021.1.5 set seed for different process'''
    if base_seed is not None:
        if local_rank != -1:
            init_seeds(base_seed + local_rank)
        else:
            init_seeds(base_seed)

    settings = ws_settings.Settings()
    settings.script_name = script_name
    settings.config_name = config_name
    settings.project_path = 'train/{}/{}'.format(script_name, config_name)   ### train/tbsi_track/vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k
    if script_name_prv is not None and config_name_prv is not None:
        settings.project_path_prv = 'train/{}/{}'.format(script_name_prv, config_name_prv)
    settings.local_rank = local_rank
    settings.save_dir = os.path.abspath(save_dir)   # /disk0/gd/home/TBSI-main/output/vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k
    settings.use_lmdb = use_lmdb
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    settings.cfg_file = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (script_name, config_name))#/disk0/gd/home/TBSI-main/experiments/tbsi_track/vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k.yaml
    settings.use_wandb = use_wandb
    if distill:
        settings.distill = distill
        settings.script_teacher = script_teacher
        settings.config_teacher = config_teacher
        if script_teacher is not None and config_teacher is not None:
            settings.project_path_teacher = 'train/{}/{}'.format(script_teacher, config_teacher)
        settings.cfg_file_teacher = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (script_teacher, config_teacher))
        expr_module = importlib.import_module('lib.train.train_script_distill')
    else:
        if script_name == 'tbsi_track':
            expr_module = importlib.import_module('lib.train.train_script')
        else:
            expr_module = importlib.import_module('lib.train.train_script_select')
    expr_func = getattr(expr_module, 'run')

    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--script', type=str, default='tbsi_track',  help='Name of the train script.')
    parser.add_argument('--config', type=str, default='vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k',  help="Name of the config file.")  ####
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--save_dir', type=str, default='./output_lr2e-4/vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k' ,help='the directory to save checkpoints and logs') ######
    parser.add_argument('--seed', type=int, default=42, help='seed for random numbers')
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--script_prv', type=str, default=None, help='Name of the train script of previous model.')
    parser.add_argument('--config_prv', type=str, default=None, help="Name of the config file of previous model.")
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], default=0)  # whether to use wandb
    # for knowledge distillation
    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)  # whether to use knowledge distillation
    parser.add_argument('--script_teacher', default=None,type=str, help='teacher script name')
    parser.add_argument('--config_teacher', default=None,type=str, help='teacher yaml configure file name')


    args = parser.parse_args()
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(0)
    run_training(args.script, args.config, cudnn_benchmark=args.cudnn_benchmark,
                local_rank=args.local_rank, save_dir=args.save_dir, base_seed=args.seed,
                use_lmdb=args.use_lmdb, script_name_prv=args.script_prv, config_name_prv=args.config_prv,
                use_wandb=args.use_wandb,
                distill=args.distill, script_teacher=args.script_teacher, config_teacher=args.config_teacher)

##  python lib/train/run_training.py --script tbsi_track --config vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --save_dir ./output/vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k 
#   --use_lmdb 0 --script_prv None --config_prv baseline --distill 0 --script_teacher None --config_teacher None --use_wandb 0

if __name__ == '__main__':
    main()
