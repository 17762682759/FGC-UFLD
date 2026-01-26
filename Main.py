# 开发时间：2023/07/03 14:06
import os
from solver import Solver
from torch.backends import cudnn


from lib.datasets import COFW, WFLW, Face300W, AFLW
from torch.utils.data import DataLoader
import yaml
from yacs.config import CfgNode as CN

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    if config.phase == 'test':
        train_loaders = None
    else:
        cofw_train_loader = DataLoader(
            dataset=COFW(config, is_train=True),
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY)
        wflw_train_loader = DataLoader(
            dataset=WFLW(config, is_train=True),
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY)
        face300w_train_loader = DataLoader(
            dataset=Face300W(config, is_train=True),
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY)
        aflw_train_loader = DataLoader(
            dataset=AFLW(config, is_train=True),
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY)
        train_loaders = {'AFLW': aflw_train_loader, 'WFLW': wflw_train_loader, '300W': face300w_train_loader,
                         'COFW': cofw_train_loader, }

    aflw_val_loader = DataLoader(
        dataset=AFLW(config, is_train=False),
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)
    face300w_val_loader = DataLoader(
        dataset=Face300W(config, is_train=False),
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    wflw_val_loader = DataLoader(
        dataset=WFLW(config, is_train=False),
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    cofw_val_loader = DataLoader(
        dataset=COFW(config, is_train=False),
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)
    val_loaders = {'AFLW': aflw_val_loader, 'WFLW': wflw_val_loader, '300W': face300w_val_loader, 'COFW': cofw_val_loader}

    # Solver for training and testing.
    solver = Solver(train_loaders, val_loaders, config)
    if config.phase == 'train':
        solver.train()
    else:
        solver.load_state_dict(config.best_model)
        solver.test()


if __name__ == '__main__':
    config_dir = "./multiData.yaml"
    with open(config_dir, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # 将字典转换为CfgNode对象
    cfg = CN(config_dict)
    main(cfg)
