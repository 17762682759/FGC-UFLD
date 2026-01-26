# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 19:07:02 2018
@author: xiang
"""

import os
import time
import datetime
from lib.utils.utils import flip_channels, shuffle_channels_for_horizontal_flipping, get_preds, draw_gaussian, \
    setup_logger
from model import FAN
from lib.utils.transforms import get_transformer_coords
from itertools import cycle

import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from AGGC import AGGC


def visualize_with_landmarks(images, targets):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    batch_size = images.shape[0]
    for i in range(batch_size):
        image = images[i]
        landmarks = targets[i]

        # 将 PyTorch 张量转换为 NumPy 数组
        image_np = image.cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))  # 从 [C, H, W] 转换为 [H, W, C]

        # 反标准化图像
        image_np = (image_np * std + mean) * 255.0
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        # RGB -> BGR（OpenCV 使用 BGR 顺序）
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 绘制特征点
        image_with_landmarks = np.ascontiguousarray(image_bgr)
        for landmark in landmarks:
            x, y = landmark[0], landmark[1]
            cv2.circle(image_with_landmarks, (int(x), int(y)), radius=1, color=(0, 255, 0), thickness=-1)

        plt.imshow(cv2.cvtColor(image_with_landmarks, cv2.COLOR_BGR2RGB))
        plt.show()


def compute_nme_ip(preds, targets):
    preds = preds.cpu()
    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i,], target[i,]
        if L == 19:  # aflw
            interocular = np.linalg.norm(pts_gt[7,] - pts_gt[10,])
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[16,] - pts_gt[17,])
        elif L == 68:  # 300w
            # interocular
            lcenter = (pts_gt[36, :] + pts_gt[37, :] + pts_gt[38, :] + pts_gt[39, :] + pts_gt[40, :] + pts_gt[41,
                                                                                                       :]) / 6
            rcenter = (pts_gt[42, :] + pts_gt[43, :] + pts_gt[44, :] + pts_gt[45, :] + pts_gt[46, :] + pts_gt[47,
                                                                                                       :]) / 6
            interpupil = np.linalg.norm(lcenter - rcenter)
            rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interpupil * L)
            continue
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[96,] - pts_gt[97,])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse


def compute_nme_io(preds, targets, meta):
    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()
    target = targets.cpu().numpy()
    global temp_sum
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i,], target[i,]
        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8,] - pts_gt[9,])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36,] - pts_gt[45,])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60,] - pts_gt[72,])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)
    return rmse


class Solver(object):
    """Solver for training fan."""

    def __init__(self, train_loader, val_loaders, config):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = train_loader
        self.val_loaders = val_loaders

        # Training configurations.
        self.nPoints = config.nPoints  # 124
        self.num_iters = config.num_iters  # 100000
        self.lr = config.lr  # 3.0e-4
        self.weightDecay = config.weightDecay  # 0
        self.resume_iters = config.resume_iters  # 0
        self.beta1 = config.beta1  # 0.5
        self.beta2 = config.beta2  # 0.999
        self.phase = config.phase  # train

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_error = 1
        # Directories.
        self.log_dir = config.log_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step  # 100
        self.model_save_step = config.model_save_step  # 500
        self.lr_update_step = config.lr_update_step  # 40000 (兼容旧字段)

        self.test_step = 0
        self.best = 0

        self._cofw_landmark_index = config.DATASET_COFW.LANDMARK_INDEX  # 29
        self._aflw_landmark_index = config.DATASET_AFLW.LANDMARK_INDEX  # 19
        self._300w_landmark_index = config.DATASET_300W.LANDMARK_INDEX  # 68
        self._wflw_landmark_index = config.DATASET_WFLW.LANDMARK_INDEX  # 98

        self.landmark_indices = {
            'COFW': self._cofw_landmark_index,
            'AFLW': self._aflw_landmark_index,
            '300W': self._300w_landmark_index,
            'WFLW': self._wflw_landmark_index
        }
        self.heatmap_size = config.MODEL.HEATMAP_SIZE  # [128, 128]
        self.image_size = config.MODEL.IMAGE_SIZE  # [128, 128]
        self.gamma = config.gamma  # 3

        # ===== LR 调度配置（新增） =====
        self.lr_gamma = getattr(config, 'lr_gamma', 0.5)  # 衰减系数
        self.lr_decay_iters = getattr(config, 'lr_decay_iters', None)  # 固定步长（按迭代）
        self.lr_milestones = getattr(config, 'lr_milestones', None)  # 里程碑（按迭代列表）

        # --- 先初始化 logger，再 build model（保证 print_network 可写日志） ---
        self.logger = setup_logger()
        self.build_model()

    def save_best_model(self, best_model_path):
        """保存最优模型（不覆盖普通 checkpoint）"""
        state = {
            'resume_iters': self.resume_iters,
            'state_dict': self.model.state_dict(),
            'optimizer': self.base_optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'lr': self.base_optimizer.param_groups[0]['lr'],
            'min_error': self.min_error
        }
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        torch.save(state, best_model_path)
        print(f'Saved BEST model to {best_model_path}')
        if hasattr(self, 'logger') and self.logger is not None:
            try:
                self.logger.info(f'Saved BEST model to {best_model_path}')
            except Exception:
                pass

    def build_model(self):
        """Create network."""
        self.model = FAN()
        self.base_optimizer = torch.optim.Adam(
            self.model.parameters(), self.lr, (self.beta1, self.beta2),
            weight_decay=self.weightDecay
        )
        self.optimizer = AGGC(self.base_optimizer)
        self.criterion = torch.nn.MSELoss()

        # 打印并写入日志：参数量
        self.print_network(self.model, 'model')

        self.model.to(self.device)

        # ================== 学习率调度：按“迭代”衰减 ==================
        decay_gamma = getattr(self, 'lr_gamma', 0.5)
        milestones = getattr(self, 'lr_milestones', None)
        decay_iters = getattr(self, 'lr_decay_iters', None)

        if milestones is not None:
            ms = sorted(list(milestones))

            def lr_lambda(iter_idx: int):
                k = 0
                while k < len(ms) and iter_idx >= ms[k]:
                    k += 1
                return (decay_gamma ** k)
        else:
            step_it = decay_iters if decay_iters is not None else self.lr_update_step
            step_it = max(int(step_it), 1)

            def lr_lambda(iter_idx: int):
                k = iter_idx // step_it
                return (decay_gamma ** k)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.base_optimizer, lr_lambda=lr_lambda
        )
        self._last_logged_lr = self.base_optimizer.param_groups[0]['lr']

        # >>> LR LOG: 初始化时打印一次当前学习率
        init_lr_msg = f'[LR] Init -> lr: {self._last_logged_lr:.6g}'
        print(init_lr_msg)
        try:
            self.logger.info(init_lr_msg)
        except Exception:
            pass
        # <<< LR LOG

    def print_network(self, model, name):
        """Print out the network information."""
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        msg = f'Total params: {total_params:.2f}M'
        # 控制台打印
        print(msg)
        # 写入日志文件
        if hasattr(self, 'logger') and self.logger is not None:
            try:
                self.logger.info(msg)
            except Exception:
                pass

    def load_state_dict(self, path_best_model):
        self.model.load_state_dict(torch.load(path_best_model))

    def save_checkpoint(self, model_path, resume_iters):
        # 保存完整快照（包含优化器与 scheduler，便于断点续训）
        state = {
            'resume_iters': resume_iters,
            'state_dict': self.model.state_dict(),
            'optimizer': self.base_optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'lr': self.base_optimizer.param_groups[0]['lr'],
            'min_error': self.min_error
        }
        torch.save(state, model_path)

    def restore_model(self):
        """Restore the trained model."""
        print('Loading the pretrained models.')
        model_path = os.path.join(self.model_save_dir, 'Checkpoint.pth.tar')
        state = torch.load(model_path)
        self.resume_iters = state.get('resume_iters', 0)
        self.model.load_state_dict(state['state_dict'])
        self.base_optimizer.load_state_dict(state['optimizer'])
        if 'scheduler' in state:
            try:
                self.scheduler.load_state_dict(state['scheduler'])
            except Exception:
                pass
        self.min_error = state.get('min_error', 1.0)
        self.lr = self.base_optimizer.param_groups[0]['lr']
        self._last_logged_lr = self.lr

        # >>> LR LOG: 恢复后打印一次当前学习率
        restore_lr_msg = f'[LR] Restore -> lr: {self.lr:.6g} (iter={self.resume_iters})'
        print(restore_lr_msg)
        try:
            self.logger.info(restore_lr_msg)
        except Exception:
            pass
        # <<< LR LOG

    def build_logger(self):
        """(未使用) 如需自定义 TensorBoard 等可在此扩展"""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, lr):
        """Manually set LR (kept for compatibility)."""
        for param_group in self.base_optimizer.param_groups:
            param_group['lr'] = lr
        if hasattr(self, 'optimizer') and hasattr(self.optimizer, 'param_groups'):
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        self.lr = lr
        self._last_logged_lr = lr
        # 同步到日志
        if hasattr(self, 'logger') and self.logger is not None:
            try:
                self.logger.info(f'[Manual LR Update] lr -> {lr:.6g}')
            except Exception:
                pass

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer.zero_grad()

    def train(self):
        """Train network."""
        global ii
        if self.resume_iters:
            self.restore_model()
            start_iters = self.resume_iters
        else:
            start_iters = 0

        print('Start training...')
        start_time = time.time()
        self.model.train()

        data_iter = [iter(self.train_loader["AFLW"]), iter(self.train_loader["WFLW"]),
                     iter(self.train_loader["300W"]), iter(self.train_loader["COFW"])]
        order = ["AFLW", "WFLW", "300W", "COFW"]

        for i in range(self.num_iters):
            self.model.train()
            self.reset_grad()
            images_mix, target_mix, target_heatmap_mix, meta_mix = [], [], [], []
            for ii in range(len(self.train_loader.keys())):
                try:
                    images, target, target_heatmap, meta = next(data_iter[ii])
                except:
                    data_iter[ii] = iter(self.train_loader[order[ii]])
                    images, target, target_heatmap, meta = next(data_iter[ii])
                images_mix.append(images)
                target_mix.append(target)
                target_heatmap_mix.append(target_heatmap)
                meta_mix.append(meta)

            images = torch.cat(images_mix, dim=0).to(self.device)
            outputs = self.model(images)

            split_outputs = torch.split(outputs, split_size_or_sections=1, dim=0)
            all_losses = []
            for j in range(4):
                outputs_alone = split_outputs[j]
                target_alone = target_mix[j].to(self.device)
                target_heatmap_mix_alone = target_heatmap_mix[j].to(self.device)
                num_joints = target_alone.shape[1]
                if num_joints == 29:
                    landmark_index = self._cofw_landmark_index
                elif num_joints == 68:
                    landmark_index = self._300w_landmark_index
                elif num_joints == 19:
                    landmark_index = self._aflw_landmark_index
                else:
                    landmark_index = self._wflw_landmark_index
                new_outputs = outputs_alone[:, landmark_index, :, :]
                loss = self.criterion(new_outputs, target_heatmap_mix_alone)
                all_losses.append(loss)

            # 反向 + 更新
            self.optimizer.pc_backward([all_losses[0], all_losses[1], all_losses[2], all_losses[3]])
            self.optimizer.step()

            # 学习率调度（按迭代）
            self.scheduler.step()

            # >>> LR LOG: 学习率变化时，打印一次（带里程碑/步长的拐点）
            cur_lr = self.base_optimizer.param_groups[0]['lr']
            if cur_lr != self._last_logged_lr:
                change_msg = f'[LR] Iter {i + 1}/{self.num_iters} -> lr changed: {self._last_logged_lr:.6g} -> {cur_lr:.6g}'
                print(change_msg)
                try:
                    self.logger.info(change_msg)
                except Exception:
                    pass
                self._last_logged_lr = cur_lr
            # <<< LR LOG

            # 训练日志（定期带上当前 LR）
            if (i + 1) % self.log_step == 0:
                if ii == 3:
                    et = (time.time() - start_time)
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    msg = ('[{0}]\t'
                           'Iter: [{1}/{2}]\t'
                           'LR: {lr:.6g}\t'  # >>> LR LOG: 将当前 LR 加入训练日志行
                           'AFLW_L: {AFLW_L:.5f} WFLW_L: {WFLW_L:.5f}  300W_L: {W300_L:.5f} COFW_L: {COFW_L:.5f}').format(
                        et, i + 1, self.num_iters,
                        lr=cur_lr,
                        AFLW_L=all_losses[0], WFLW_L=all_losses[1],
                        W300_L=all_losses[2], COFW_L=all_losses[3]
                    )
                    print(msg)
                    self.logger.info(msg)

            # 保存 checkpoint + 测试
            if (i + 1) % self.model_save_step == 0:
                model_save_path = os.path.join(self.model_save_dir, 'Checkpoint.pth.tar')
                self.save_checkpoint(model_save_path, i + 1)
                save_msg = 'Save model checkpoint into {}...'.format(self.model_save_dir)
                print(save_msg)
                self.logger.info(save_msg)

                self.test()

            # 旧的按 lr_update_step 手动衰减逻辑已由 scheduler 接管
            # if (i + 1) % self.lr_update_step == 0: ...

    def test(self):
        datasets = ["300W", "AFLW", "WFLW", "COFW"]
        for ds in datasets:
            nme_ip, nme_io = self.validate(ds)

            print(f"{ds}: nme_ip={nme_ip:.5f}, nme_io={nme_io:.5f}")
            self.logger.info(f"{ds}: nme_ip={nme_ip:.5f}, nme_io={nme_io:.5f}")

            # 保存每个数据集的 best 模型
            min_error_attr = f"min_error_{ds.lower()}"
            if not hasattr(self, min_error_attr):
                setattr(self, min_error_attr, 1.0)

            if nme_ip < getattr(self, min_error_attr):
                setattr(self, min_error_attr, nme_ip)
                best_model_path = os.path.join(self.model_save_dir, f"best_model/BestModel_{ds}.pth.tar")
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                self.save_checkpoint(best_model_path, self.resume_iters)
                print(f"[Best Model Saved] {ds}: nme_ip improved to {nme_ip:.5f} -> {best_model_path}")
                self.logger.info(f"[Best Model Saved] {ds}: nme_ip improved to {nme_ip:.5f} -> {best_model_path}")

    def validate(self, dataset):
        self.model.eval()
        val_loader = self.val_loaders[dataset]
        nme_batch_sum_ip = 0
        nme_batch_sum_io = 0
        nme_count = 0

        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                input, targets, target_weight, meta = batch
                device = self.device
                input = input.to(device)

                num_joints = targets.shape[1]
                if num_joints == 29:
                    landmark_index = self._cofw_landmark_index
                elif num_joints == 68:
                    landmark_index = self._300w_landmark_index
                elif num_joints == 98:
                    landmark_index = self._wflw_landmark_index
                elif num_joints == 19:
                    landmark_index = self._aflw_landmark_index
                else:
                    raise TypeError(f"No match for points {num_joints}")

                out_heatmap1 = self.model(input)[:, landmark_index, :, :]

                images_flip = torch.from_numpy(input.cpu().numpy()[:, :, :, ::-1].copy()).to(device)
                out_heatmap2 = self.model(images_flip)[:, landmark_index, :, :]
                out_heatmap2 = flip_channels(out_heatmap2.cpu())
                out_heatmap2 = shuffle_channels_for_horizontal_flipping(out_heatmap2)
                out_heatmap = (out_heatmap1.cpu() + out_heatmap2) / 2

                pred_coords = 2 * get_preds(out_heatmap)

                nme_count += targets.shape[0]

                pred_coords = get_transformer_coords(pred_coords, meta, torch.tensor(
                    [meta['output_size'][0][0], meta['output_size'][1][0]]))
                targets = get_transformer_coords(targets, meta,
                                                 torch.tensor([meta['output_size'][0][0], meta['output_size'][1][0]]))

                nme_batch_sum_ip += np.sum(compute_nme_ip(pred_coords, targets))
                nme_batch_sum_io += np.sum(compute_nme_io(pred_coords, targets, meta))

        nme_ip = nme_batch_sum_ip / nme_count
        nme_io = nme_batch_sum_io / nme_count
        return nme_ip, nme_io


