'''
OMP_NUM_THREADS=2 python3 -m torch.distributed.launch --nproc_per_node=2 main.py
'''

import argparse
import os
import random
import shutil
from datetime import datetime
from pprint import pprint

import numpy as np
import toml
import torch
from torch.utils.data import DataLoader

import utils
from dataloader.data_generator import DataGenerator
from dataloader.image_file import ImageFileTrain, ImageFileTest
from dataloader.prefetcher import Prefetcher
from trainers.trainer import Trainer
from utils import CONFIG

torch.manual_seed(8282)
torch.cuda.manual_seed(8282)
torch.cuda.manual_seed_all(8282) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(8282)
random.seed(8282)


def copy_script(root_path=None):
    if not os.path.exists(root_path):
        os.makedirs(root_path)
        os.makedirs(CONFIG.log.logging_path)
        os.makedirs(CONFIG.log.checkpoint_path)

    shutil.copytree('./config', os.path.join(root_path, 'config'), ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree('./dataloader', os.path.join(root_path, 'dataloader'), ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree('./networks', os.path.join(root_path, 'networks'), ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree('./trainers', os.path.join(root_path, 'trainers'), ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree('./utils', os.path.join(root_path, 'utils'), ignore=shutil.ignore_patterns('__pycache__'))

    shutil.copy('./main.py', os.path.join(root_path, 'main.py'))
    shutil.copy('./inference.py', os.path.join(root_path, 'inference.py'))
    shutil.copy('./evaluation.py', os.path.join(root_path, 'evaluation.py'))


def main():
    # Train or Test
    if CONFIG.phase.lower() == "train":
        # set distributed training
        if CONFIG.dist:
            CONFIG.gpu = CONFIG.local_rank
            torch.cuda.set_device(CONFIG.gpu)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            CONFIG.world_size = torch.distributed.get_world_size()

        # Create directories if not exist.
        if CONFIG.local_rank == 0:
            utils.make_dir(CONFIG.log.logging_path)
            utils.make_dir(CONFIG.log.checkpoint_path)

        """=== Set logger ==="""
        logger = utils.get_logger(CONFIG.log.logging_path, logging_level=CONFIG.log.logging_level)

        """=== Set data loader ==="""
        # [1] Composition-1k dataset
        train_image_file = ImageFileTrain(alpha_dir=CONFIG.data.train_alpha,
                                          fg_dir=CONFIG.data.train_fg,
                                          bg_dir=CONFIG.data.train_bg)
        test_image_file = ImageFileTest(alpha_dir=CONFIG.data.test_alpha,
                                        merged_dir=CONFIG.data.test_merged,
                                        trimap_dir=CONFIG.data.test_trimap)
        train_dataset = DataGenerator(train_image_file, phase='train')
        test_dataset = DataGenerator(test_image_file, phase='val')

        if CONFIG.dist:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            train_sampler = None
            test_sampler = None

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=CONFIG.model.batch_size,
                                      shuffle=(train_sampler is None),
                                      num_workers=CONFIG.data.workers,
                                      pin_memory=True,
                                      sampler=train_sampler,
                                      drop_last=True)
        train_dataloader = Prefetcher(train_dataloader)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=CONFIG.data.workers,
                                     sampler=test_sampler,
                                     drop_last=False)

        """=== Set Trainer ==="""
        trainer = Trainer(train_dataloader=train_dataloader,
                          test_dataloader=test_dataloader,
                          logger=logger)
        """=== Run Trainer ==="""
        trainer.train()

    else:
        raise NotImplementedError("Unknown Phase: {}".format(CONFIG.phase))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--local_rank', type=int, default=0)

    # Composition-1k
    parser.add_argument('--config', type=str, default='config/MatteFormer_Composition1k.toml')

    # Parse configuration
    args = parser.parse_args()
    with open(args.config) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")
    CONFIG.phase = args.phase

    # set_experiment path
    CONFIG.log.experiment_root = os.path.join(CONFIG.log.experiment_root, datetime.now().strftime("%y%m%d_%H%M%S"))
    CONFIG.log.logging_path = os.path.join(CONFIG.log.experiment_root, CONFIG.log.logging_path)
    CONFIG.log.checkpoint_path = os.path.join(CONFIG.log.experiment_root, CONFIG.log.checkpoint_path)

    if args.local_rank == 0:
        print('CONFIG: ')
        pprint(CONFIG)
        copy_script(root_path=CONFIG.log.experiment_root)

    CONFIG.local_rank = args.local_rank

    # Train
    main()

