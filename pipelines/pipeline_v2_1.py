'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os
from tqdm import tqdm
import shutil
import time
from torch.utils.data import Subset

# Ingnore all numba warning
# from numba.core.errors import NumbaWarning
# import warnings
# warnings.simplefilter('ignore', category=NumbaWarning)

import logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.ERROR)

from torch.utils.tensorboard import SummaryWriter

from utils.util_pipeline import *
from utils.util_point_cloud import *
from utils.util_config import cfg, cfg_from_yaml_file

import nms
from utils.util_point_cloud import Object3D
import utils.kitti_eval.kitti_common as kitti
from utils.kitti_eval.eval import get_official_eval_result

class Pipeline_v2_1():
    def __init__(self, path_cfg=None, split='train', mode='train/val'):
        '''
        * split = 'train' or 'test'
        * mode = 'train', 'train/val', 'val', 'test'
        '''
        if (mode=='val') or (mode=='train/val'):
            self.cfg = cfg_from_yaml_file(path_cfg, cfg)
            self.modify_cfg(mode)
            self.dataset_val = build_dataset(self, 'test')

        self.cfg = cfg_from_yaml_file(path_cfg, cfg)
        self.modify_cfg(mode)


        if self.cfg.GENERAL.SEED is not None:
            set_random_seed(cfg.GENERAL.SEED)
        
        self.dataset = build_dataset(self, split=split)
        self.cfg.DATASET.NUM = len(self.dataset)
        
        # cube or tesseract, pick just one
        if (self.cfg.DATASET.GET_ITEM['rdr_cube']):
            self.get_physical_values(dtype='cube')
            self.update_cfg(dtype='cube')
        elif (self.cfg.DATASET.GET_ITEM['rdr_tesseract']):
            self.get_physical_values(dtype='tesseract')
            self.update_cfg(dtype='tesseract')

        self.network = build_network(self).cuda()

        self.epoch_start = 0
        
        self.optimizer = build_optimizer(self, self.network)
        self.scheduler = build_scheduler(self, self.optimizer)

        if self.cfg.GENERAL.IS_TRAIN and \
            self.cfg.GENERAL.LOGGING.IS_LOGGING:
            self.set_logging(path_cfg)
        else:
            self.is_logging = False

        if self.cfg.VAL.IS_VALIDATE:
            self.set_validate()
        else:
            self.is_validate = False
        
        if self.cfg.GENERAL.RESUME.IS_RESUME:
            self.resume_network()
        
        # self.pline_description()

    def modify_cfg(self, mode):
        if mode == 'test':
            self.cfg.GENERAL.LOGGING.IS_LOGGING = False
            self.cfg.OPTIMIZER.NUM_WORKERS = 1
            self.cfg.MODEL.BACKBONE.PRETRAINED = False
        return

    def update_cfg(self, dtype='tesseract'):
        if dtype == 'cube':
            self.cfg.DATASET.RDR_CUBE.ARR_Z = self.arr_z_cb.copy()
            self.cfg.DATASET.RDR_CUBE.ARR_Y = self.arr_y_cb.copy()
            self.cfg.DATASET.RDR_CUBE.ARR_X = self.arr_x_cb.copy()
        elif dtype == 'tesseract':
            len_doppler, len_range, len_azimuth, len_elevation = \
                                    self.dataset[0]['rdr_tesseract'].shape
            self.cfg.MODEL.DRAE_SIZE = \
                    [len_doppler, len_range, len_azimuth, len_elevation]

            self.cfg.MODEL.ARR_RANGE = self.arr_range.copy()
            self.cfg.MODEL.ARR_AZIMUTH = self.arr_azimuth.copy()
            self.cfg.MODEL.ARR_ELEVATION = self.arr_elevation.copy()

    def set_validate(self):
        self.is_validate = True
        self.is_consider_subset = self.cfg.VAL.IS_CONSIDER_VAL_SUBSET
        self.val_per_epoch_subset = self.cfg.VAL.VAL_PER_EPOCH_SUBSET
        self.val_num_subset = self.cfg.VAL.NUM_SUBSET
        self.val_per_epoch_full = self.cfg.VAL.VAL_PER_EPOCH_FULL
        self.val_cls_pred = self.cfg.VAL.LIST_CLS_PRED
        self.val_cls_label = self.cfg.VAL.LIST_CLS_LABEL
        self.list_val_conf_thr = self.cfg.VAL.LIST_VAL_CONF_THR
        self.list_care_cls_idx = self.cfg.VAL.LIST_CLS_CARE
        self.val_iou_mode = self.cfg.VAL.VAL_IOU_MODE
    
    def pline_description(self):
        print('* newtork (description start) -------')
        print(self.network)
        print('* newtork (description end) ---------')
        print('* optimizer (description start) -----')
        print(self.optimizer)
        print('* optimizer (description end) -------')

        len_data = len(self.dataset)
        print(f'* dataset length = {len_data}')

        try:
            shp_tesseract = self.dataset[0]['rdr_tesseract'].shape
            print(f'* shape tesseract = {shp_tesseract}')
        except:
            print('* no tesseract')

    def get_physical_values(self, dtype='tessseract', is_show_values=False):
        if dtype == 'cube':
            self.arr_z_cb = self.dataset.arr_z_cb
            self.arr_y_cb = self.dataset.arr_y_cb
            self.arr_x_cb = self.dataset.arr_x_cb

        elif dtype == 'tesseract':
            self.arr_range = self.dataset.arr_range
            self.arr_azimuth = self.dataset.arr_azimuth
            self.arr_elevation = self.dataset.arr_elevation

            if is_show_values:
                print(self.arr_range)
                print((np.array(self.arr_azimuth)*180./np.pi).tolist())
                print((np.array(self.arr_elevation)*180./np.pi).tolist())

                print(len(self.arr_range))
                print(len(self.arr_azimuth))
                print(len(self.arr_elevation))
    
    def set_logging(self, path_cfg, is_print_where=True):
        self.is_logging = True
        str_local_time = get_local_time_str()
        str_exp = self.cfg.MODEL.NAME + '_' + str_local_time
        self.path_log = os.path.join(self.cfg.GENERAL.LOGGING.PATH_LOGGING, str_exp)

        if is_print_where:
            print(f'Start logging in {str_exp} ...')

        if not (os.path.exists(self.path_log)):
            os.makedirs(self.path_log)

        self.list_key_logging = self.cfg.GENERAL.LOGGING.LIST_KEY_LOGGING
        self.log_train_iter = SummaryWriter(os.path.join(self.path_log, 'train_iter'), comment='train_iter')
        self.log_train_epoch = SummaryWriter(os.path.join(self.path_log, 'train_epoch'), comment='train_epoch')
        self.log_val = SummaryWriter(os.path.join(self.path_log, 'val'), comment='val')
        self.log_iter_start = None

        # graph loading (TBD) # https://www.youtube.com/watch?v=74aSImrIEbQ

        self.is_save_model = self.cfg.GENERAL.LOGGING.IS_SAVE_MODEL
        if self.is_save_model:
            os.makedirs(os.path.join(self.path_log, 'models'))
            os.makedirs(os.path.join(self.path_log, 'utils'))

        # cfg backup (same files, just for identification)
        name_file_origin = path_cfg.split('/')[-1] # original cfg file name
        name_file_cfg = 'config.yml'
        shutil.copy2(path_cfg, os.path.join(self.path_log, name_file_origin))
        shutil.copy2(path_cfg, os.path.join(self.path_log, name_file_cfg))

    def resume_network(self):
        path_exp = self.cfg.GENERAL.RESUME.PATH_EXP
        path_state_dict = os.path.join(path_exp, 'utils')
        epoch = self.cfg.GENERAL.RESUME.START_EP
        list_epochs = sorted(list(map(lambda x: int(x.split('.')[0].split('_')[1]), os.listdir(path_state_dict))))
        # print(list_epochs)
        epoch = list_epochs[-1] if epoch is None else epoch

        path_state_dict = os.path.join(path_state_dict, f'util_{epoch}.pt')
        print('* Start resume, path_state_dict =  ', path_state_dict)
        state_dict = torch.load(path_state_dict)

        try:
            self.epoch_start = epoch + 1
            self.network.load_state_dict(state_dict['model_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.log_iter_start = state_dict['idx_log_iter']
            print(f'* Network & Optimizer are loaded / Resume epoch is {epoch} / Start from {self.epoch_start} ...')
        except:
            raise AttributeError('* ')

        if ('scheduler_state_dict' in state_dict.keys()) and (not (self.scheduler is None)):
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            print('* Scheduler is loaded ...')
        else:
            print('* Scheduler is started from vanilla ...')

        ### Copy tree logging! ###
        list_copy_dirs = ['train_epoch', 'train_iter', 'val', 'val_kitti']
        if (self.cfg.GENERAL.RESUME.IS_COPY_LOGS) and (self.is_logging):
            for copy_dir in list_copy_dirs:
                shutil.copytree(os.path.join(path_exp, copy_dir), \
                    os.path.join(self.path_log, copy_dir), dirs_exist_ok=True)
        ### Copy tree logging! ###

        return

    def preprocess_kradar(self):
        self.dataset.preprocess_sparse_tensor(cfg)

    def preprocess_cfar(self, save_dir, folder_name):
        self.dataset.preprocess_sparse_tensor_cfar(cfg, save_dir, folder_name)
    

    def train_network(self, is_shuffle=True):
        self.network.train()
        # t1 = time.time()
        if cfg.OPTIMIZER.BATCH_SIZE == 1:
            data_loader_train = torch.utils.data.DataLoader(self.dataset, \
                batch_size = self.cfg.OPTIMIZER.BATCH_SIZE, shuffle = is_shuffle)
        else:
            data_loader_train = torch.utils.data.DataLoader(self.dataset, \
                batch_size = self.cfg.OPTIMIZER.BATCH_SIZE, shuffle = is_shuffle, \
                collate_fn = self.dataset.collate_fn, num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
        # t2 = time.time()
        # print(f"* 4: {t2 - t1:.5f} sec")

        epoch_start = self.epoch_start
        epoch_end = self.cfg.OPTIMIZER.MAX_EPOCH

        # Sanity check for validation
        # self.validate_kitti(0, list_conf_thr=self.list_val_conf_thr, is_subset=True)



        ##########################################
        if self.is_logging:
            idx_log_iter = 0 if self.log_iter_start is None else self.log_iter_start

        for epoch in range(epoch_start, epoch_end):
            print(f'* Training epoch = {epoch}/{epoch_end-1}')
            if self.is_logging:
                print(f'* Logging path = {self.path_log}')
            self.network.train()
            self.network.training = True
            idx_fails = []
            num_fails = 0
            avg_loss = []
            for idx_iter, dict_datum in enumerate(tqdm(data_loader_train)):
                ### Debug ###
                # if idx_iter < 35:
                #     continue
                ### Debug ###
                # # # try:
                # # # if dict_datum is None:
                # # #     # print('* dataloader error...')
                # # #     continue

                # # # try:
                # # #     if (idx_iter == len(data_loader_train)-1):
                # # #         continue

                    # print(dict_datum['meta'][0])

                    # t1 = time.time()
                    dict_net = self.network(dict_datum)

                    # t2 = time.time()
                    # print(f"* network: {t2 - t1:.5f} sec")

                    # t1 = time.time()
                    loss = self.network.head.loss(dict_net)

                    if hasattr(self.network, 'point_head'): # PVRCNN_PP
                        point_loss = self.network.point_head.loss(dict_net)
                        loss += point_loss

                    if hasattr(self.network, 'roi_head'): # PVRCNN_PP
                        roi_loss = self.network.roi_head.loss(dict_net)
                        loss += roi_loss

                    # t2 = time.time()
                    # print(f"* loss calculation: {t2 - t1:.5f} sec")
                    
                    try:
                        log_avg_loss = loss.cpu().detach().item()
                    except:
                        log_avg_loss = loss
                    avg_loss.append(log_avg_loss)

                    # t1 = time.time()
                    if loss == 0.:
                        print('loss is 0.') # no label
                    elif torch.isfinite(loss):
                        loss.backward()
                    else:
                        # raise TypeError('Nan or inf loss happend !')
                        print('>>> Nan or inf loss happend !')
                        print(dict_datum['meta'])
                        print(dict_datum['labels'])
                    self.optimizer.step()
                    if not (self.scheduler is None):
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    # t2 = time.time()
                    # print(f"* optimization: {t2 - t1:.5f} sec")

                    # t1 = time.time()
                    if self.is_logging:
                        for key_logging in self.list_key_logging:
                            dict_temp = dict_net[key_logging]
                            idx_log_iter +=1

                            for k, v in dict_temp.items():
                                self.log_train_iter.add_scalar(f'{key_logging}/{k}', v, idx_log_iter)
                        if not (self.scheduler is None):
                            lr = self.scheduler.get_last_lr()
                            self.log_train_iter.add_scalar(f'train/learning_rate', lr[0], idx_log_iter)
                        
                # t2 = time.time()
                # print(f"* logging: {t2 - t1:.5f} sec")

                # # # except Exception as e:
                # # #     print(e)
                # # #     idx_fails.append(idx_iter)
                # # #     num_fails += 1

            if self.is_save_model:
                path_dict_model = os.path.join(self.path_log, 'models', f'model_{epoch}.pt')
                path_dict_util = os.path.join(self.path_log, 'utils', f'util_{epoch}.pt')
                torch.save(self.network.state_dict(), path_dict_model)
                dict_util = {
                    'epoch': epoch,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'idx_log_iter': idx_log_iter, 
                }
                if not (self.scheduler is None):
                    dict_util.update({'scheduler_state_dict': self.scheduler.state_dict()})
                torch.save(dict_util, path_dict_util)

            if self.is_logging:
                self.log_train_epoch.add_scalar(f'train/avg_loss', np.mean(avg_loss), epoch)
            
            # print('Average loss: ', np.mean(avg_loss))
            # print('Fails: ', num_fails, ' times')
            # print('Fail file names: ', idx_fails)

            if self.is_validate:
                self.network.training=False
                if self.is_consider_subset:
                    if ((epoch + 1) % self.val_per_epoch_subset) == 0:
                        self.validate_kitti(epoch, list_conf_thr=self.list_val_conf_thr, is_subset=True)
                if ((epoch + 1) % self.val_per_epoch_full) == 0:
                    self.validate_kitti(epoch, list_conf_thr=self.list_val_conf_thr)

    def load_dict_model(self, path_dict_model, is_strict=False):
        pt_dict_model = torch.load(path_dict_model)
        self.network.load_state_dict(pt_dict_model, strict=is_strict)

    def vis_infer_cube(self, sample_indices, conf_thr=0.1):
        subset = Subset(self.dataset, sample_indices)

        data_loader = torch.utils.data.DataLoader(subset, \
                batch_size = 1, shuffle = False, collate_fn = self.dataset.collate_fn, num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
        
        ### Assume Batch size = 1 ###
        for dict_datum in data_loader:
            dict_out = self.network(dict_datum)
            dict_out = self.network.list_modules[-1].get_pred_boxes_nms_for_single_datum(dict_out, conf_thr)

            pc_lidar = dict_datum['ldr_pc_64']

            pred_objects = []
            
            if 'pred_boxes_nms' in dict_out.keys():
                for pred_idx, pred_box in enumerate(dict_out['pred_boxes_nms']):
                    # print(pred_box.shape)
                    score, xc, yc, zc, xl, yl, zl, rot = pred_box
                    # We need class id
                    # KITTI Example: Car -1 -1 -4.2780 668.7884 173.1068 727.8801 198.9699 1.4607 1.7795 4.5159 5.3105 1.4764 43.1853 -4.1569 0.9903
                    # cls_idx = dict_out['pred_cls_ids'][pred_idx].item()
                    # cls_id = dict_out.val_cls_pred[cls_idx] # 'Car' # just change cls id

                    obj = Object3D(xc.item(), yc.item(), zc.item(), xl.item(), yl.item(), zl.item(), rot.item())
                    pred_objects.append(obj)

            ################## Processing Labels ####################
            labels = dict_out['labels'][0]
            # print(labels)
            # print(labels.shape)
            # Make corners for each object
            gt_objects = []
            for gt_object in labels:
                cls_txt, cls_id, shp, obj_idx = gt_object
                xc, yc, zc, rot_deg, xl, yl, zl = shp
                try:
                    xc, yc, zc, rot_rad, xl, yl, zl = xc.item(), yc.item(), zc.item(), np.deg2rad(rot_deg.item()), xl.item(), yl.item(), zl.item()       
                except:
                    xc, yc, zc, rot_rad, xl, yl, zl = xc, yc, zc, rot_deg, xl, yl, zl

                obj = Object3D(xc, yc, zc, xl, yl, zl, rot_rad)
                gt_objects.append(obj)

            ################# Making Bounding Boxes for Predictions and Labels ################
            # Make lines for each object
            lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                    [4, 5], [5, 6], [6, 7], [4, 7],
                    [0, 4], [1, 5], [2, 6], [3, 7],
                    [0, 2], [1, 3], [4, 6], [5, 7]]
            colors_gt = [[0, 1, 0] for _ in range(len(lines))]
            colors_pred = [[1, 0, 0] for _ in range(len(lines))]

            line_sets_gt = []
            line_sets_pred = []
            
            for gt_obj in gt_objects:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(gt_obj.corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors_gt)
                line_sets_gt.append(line_set)

            for pred_obj in pred_objects:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(pred_obj.corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors_pred)
                line_sets_pred.append(line_set)

            ################### Visualization ####################
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])
            o3d.visualization.draw_geometries([pcd] + line_sets_gt + line_sets_pred)

    def validate_kitti(self, epoch=None, list_conf_thr=None, is_subset=False):
        self.network.eval()

        ### Check is_validate with small dataset ###
        if is_subset:
            is_shuffle = True
            tqdm_bar = tqdm(total=self.val_num_subset, desc='val sub: ')
            log_header = 'val_sub'
        else:
            is_shuffle = False
            tqdm_bar = tqdm(total=len(self.dataset_val), desc='val tot: ')
            log_header = 'val_tot'

        data_loader = torch.utils.data.DataLoader(self.dataset_val, \
                batch_size = 1, shuffle = is_shuffle, collate_fn = self.dataset.collate_fn, \
                    num_workers = 1) # self.cfg.OPTIMIZER.NUM_WORKERS)
        
        if epoch is None:
            path_epoch = 'temp'
        else:
            path_epoch = f'ep_{epoch}' if is_subset else f'ep_{epoch}_tot'

        if self.cfg.VAL.DIR is None:
            path_dir = os.path.join(self.path_log, 'val_kitti', path_epoch)
        else:
            path_dir = os.path.join(self.cfg.VAL.DIR, 'val_kitti', path_epoch)
        
        for conf_thr in list_conf_thr:
            os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)
            with open(path_dir + f'/{conf_thr}/' + 'val.txt', 'w') as f:
                f.write('')

        for idx_datum, dict_datum in enumerate(data_loader):
            if is_subset & (idx_datum >= self.val_num_subset):
                break
            
            try:
                dict_out = self.network(dict_datum)
                idx_name = str(idx_datum).zfill(6)

                ### for every conf in list_conf_thr ###
                for conf_thr in list_conf_thr:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', 'desc')
                    list_dir = [preds_dir, labels_dir, desc_dir]
                    split_path = path_dir + f'/{conf_thr}/' + 'val.txt'
                    for temp_dir in list_dir:
                        os.makedirs(temp_dir, exist_ok=True)

                    dict_out = self.network.list_modules[-1].get_pred_boxes_nms_for_single_datum(dict_out, conf_thr)
                    if dict_out is None:
                        continue

                    dict_out = dict_datum_to_kitti(self, dict_out)

                    if len(dict_out['kitti_labels']) == 0: # not eval emptry label
                        # with open(labels_dir + '/' + idx_name + '.txt', mode) as f:
                        #     f.write('\n')
                        pass
                    else:
                        for idx_label, label in enumerate(dict_out['kitti_labels']):
                            if idx_label == 0:
                                mode = 'w'
                            else:
                                mode = 'a'

                            with open(labels_dir + '/' + idx_name + '.txt', mode) as f:
                                f.write(label+'\n')

                        ### Process description ###
                        with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                            f.write(dict_out['kitti_desc'])
                        ### Process description ###

                        if len(dict_out['kitti_preds']) == 0:
                            with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                                f.write('\n')
                        else:
                            for idx_pred, pred in enumerate(dict_out['kitti_preds']):
                                if idx_pred == 0:
                                    mode = 'w'
                                else:
                                    mode = 'a'

                                with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                                    f.write(pred+'\n')

                        str_log = idx_name + '\n'
                        with open(split_path, 'a') as f:
                            f.write(str_log)
                tqdm_bar.update(1)

            except Exception as e:
                print(e)

        tqdm_bar.close()

        ### Validate per conf ###
        for conf_thr in list_conf_thr:
            preds_dir = os.path.join(path_dir, f'{conf_thr}', 'preds')
            labels_dir = os.path.join(path_dir, f'{conf_thr}', 'gts')
            desc_dir = os.path.join(path_dir, f'{conf_thr}', 'desc')
            split_path = path_dir + f'/{conf_thr}/' + 'val.txt'

            dt_annos = kitti.get_label_annos(preds_dir)
            val_ids = read_imageset_file(split_path)
            gt_annos = kitti.get_label_annos(labels_dir, val_ids)
            if self.val_iou_mode == 'all':
                list_metrics = []
                list_results = []
                for idx_cls_val in self.list_care_cls_idx:
                    try:
                        dict_metrics, result = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                    except:
                        dic_cls_val = self.cfg.VAL.DIC_CLASS_VAL
                        cls_name = dic_cls_val[list(dic_cls_val.keys())[idx_cls_val]]
                        dict_metrics = {
                            'cls': cls_name,
                            'iou': self.cfg.VAL.LIST_VAL_IOU,
                            'bbox': [0., 0., 0.],
                            'bev': [0., 0., 0.],
                            '3d': [0., 0., 0.],
                        }
                        result = f'error occurs for {cls_name}\n'
                    list_metrics.append(dict_metrics)
                    list_results.append(result)

                for dict_metrics, result in zip(list_metrics, list_results):
                    cls_name = dict_metrics['cls']
                    ious = dict_metrics['iou']
                    bevs = dict_metrics['bev']
                    ap3ds = dict_metrics['3d']
                    self.log_val.add_scalars(f'{log_header}/bev_conf_{conf_thr}', {
                        f'iou_{ious[0]}_{cls_name}': bevs[0],
                        f'iou_{ious[1]}_{cls_name}': bevs[1],
                        f'iou_{ious[2]}_{cls_name}': bevs[2],
                    }, epoch)
                    self.log_val.add_scalars(f'{log_header}/3d_conf_{conf_thr}', {
                        f'iou_{ious[0]}_{cls_name}': ap3ds[0],
                        f'iou_{ious[1]}_{cls_name}': ap3ds[1],
                        f'iou_{ious[2]}_{cls_name}': ap3ds[2],
                    }, epoch)
                    self.log_val.add_text(f'{log_header}/conf_{conf_thr}_{cls_name}', result, epoch)
            elif self.val_iou_mode == 'each':

                list_iou_mode = ['hard', 'mod', 'easy']
                for idx_cls_val in self.list_care_cls_idx:
                    log_result = ''
                    for idx_mode, iou_mode in enumerate(list_iou_mode):
                        try:
                            dict_metrics, result = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, \
                                                                                iou_mode=iou_mode, is_return_with_dict=True)
                        except:
                            dic_cls_val = self.cfg.VAL.DIC_CLASS_VAL
                            cls_name = dic_cls_val[list(dic_cls_val.keys())[idx_cls_val]]
                            dict_metrics = {
                                'cls': cls_name,
                                'iou': [self.cfg.VAL.LIST_VAL_IOU[idx_mode]],
                                'bbox': [0.],
                                'bev': [0.],
                                '3d': [0.],
                            }
                            result = f'error occurs for {cls_name} in iou {iou_mode}...\n'

                        cls_name = dict_metrics['cls']
                        ious = dict_metrics['iou']
                        bevs = dict_metrics['bev']
                        ap3ds = dict_metrics['3d']
                        
                        self.log_val.add_scalars(f'{log_header}/bev_conf_{conf_thr}', {
                            f'{cls_name}_iou_{ious[0]}': bevs[0]
                        }, epoch)
                        self.log_val.add_scalars(f'{log_header}/3d_conf_{conf_thr}', {
                            f'{cls_name}_iou_{ious[0]}': ap3ds[0]
                        }, epoch)
                        log_result += result
                    self.log_val.add_text(f'{log_header}/conf_{conf_thr}_{cls_name}', log_result, epoch)
        ### Validate per conf ###


    def validate_kitti_conditional(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False):
            self.network.eval()
            road_cond_list = ['urban', 'highway', 'countryside', 'alleyway', 'parkinglots', 'shoulder', 'mountain', 'university']
            time_cond_list = ['day', 'night']
            weather_cond_list = ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow']

            ### Check is_validate with small dataset ###
            if is_subset:
                is_shuffle = True
                tqdm_bar = tqdm(total=self.val_num_subset, desc='val sub: ')
                log_header = 'val_sub'
            else:
                is_shuffle = False
                tqdm_bar = tqdm(total=len(self.dataset_val), desc='val tot: ')
                log_header = 'val_tot'

            data_loader = torch.utils.data.DataLoader(self.dataset_val, \
                    batch_size = 1, shuffle = is_shuffle, collate_fn = self.dataset.collate_fn, \
                        num_workers = 1) # self.cfg.OPTIMIZER.NUM_WORKERS)
            
            if epoch is None:
                path_epoch = 'temp'
            else:
                path_epoch = f'ep_{epoch}' if is_subset else f'ep_{epoch}_tot'

            if self.cfg.VAL.DIR is None:
                path_dir = os.path.join(self.path_log, 'val_kitti', path_epoch)
            else:
                path_dir = os.path.join(self.cfg.VAL.DIR, 'val_kitti', path_epoch)
            
            # Dirs per conf_thr
            for conf_thr in list_conf_thr:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)
                # with open(path_dir + f'/{conf_thr}/' + 'val.txt', 'w') as f:
                #     f.write('')

                ########## Dirs per conditions #############
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', 'all'), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + 'all/val.txt', 'w') as f:
                    f.write('')

                for road_cond in road_cond_list:
                    os.makedirs(os.path.join(path_dir, f'{conf_thr}', road_cond), exist_ok=True)
                    with open(path_dir + f'/{conf_thr}/' + road_cond + '/val.txt', 'w') as f:
                        f.write('')

                for time_cond in time_cond_list:
                    os.makedirs(os.path.join(path_dir, f'{conf_thr}', time_cond), exist_ok=True)
                    with open(path_dir + f'/{conf_thr}/' + time_cond + '/val.txt', 'w') as f:
                        f.write('')

                for weather_cond in weather_cond_list:
                    os.makedirs(os.path.join(path_dir, f'{conf_thr}', weather_cond), exist_ok=True)
                    with open(path_dir + f'/{conf_thr}/' + weather_cond + '/val.txt', 'w') as f:
                        f.write('')
                #############################################

                pred_dir_list = []
                label_dir_list = []
                desc_dir_list = []
                split_path_list = []

                ### For All Conditions ###
                preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)
                                
                ### For Specific Conditions
                for road_cond in road_cond_list:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'desc')
                    list_dir = [preds_dir, labels_dir, desc_dir]
                    split_path = path_dir + f'/{conf_thr}/' + road_cond +'/val.txt'
                    
                    for temp_dir in list_dir:
                        os.makedirs(temp_dir, exist_ok=True)
                    
                    pred_dir_list.append(preds_dir)
                    label_dir_list.append(labels_dir)
                    desc_dir_list.append(desc_dir)
                    split_path_list.append(split_path)
                
                for time_cond in time_cond_list:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'desc')
                    list_dir = [preds_dir, labels_dir, desc_dir]
                    split_path = path_dir + f'/{conf_thr}/' + time_cond +'/val.txt'
                    
                    for temp_dir in list_dir:
                        os.makedirs(temp_dir, exist_ok=True)

                    pred_dir_list.append(preds_dir)
                    label_dir_list.append(labels_dir)
                    desc_dir_list.append(desc_dir)
                    split_path_list.append(split_path)
                
                for weather_cond in weather_cond_list:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'desc')
                    list_dir = [preds_dir, labels_dir, desc_dir]
                    split_path = path_dir + f'/{conf_thr}/' + weather_cond +'/val.txt'
                    
                    for temp_dir in list_dir:
                        os.makedirs(temp_dir, exist_ok=True)

                    pred_dir_list.append(preds_dir)
                    label_dir_list.append(labels_dir)
                    desc_dir_list.append(desc_dir)
                    split_path_list.append(split_path)
            
                # for split_path_ in split_path_list:
                #     with open(split_path_, 'w') as f:
                #         f.write('') #initialize val txts

            # Creating gts and preds txt files for evaluation
            for idx_datum, dict_datum in enumerate(data_loader):
                if is_subset & (idx_datum >= self.val_num_subset):
                    break
                try:
                    dict_out = self.network(dict_datum)
                except:
                    print(f'error happens in {idx_datum}')
                    continue

                if is_print_memory:
                    print('max_memory: ', torch.cuda.max_memory_allocated(device=None))
                    
                idx_name = str(idx_datum).zfill(6)
                
                road_cond_tag, time_cond_tag, weather_cond_tag = dict_out['desc'][0]['road_type'], dict_out['desc'][0]['capture_time'], dict_out['desc'][0]['climate']
                # print(dict_out['desc'][0])

                ### for every conf in list_conf_thr ###
                for conf_thr in list_conf_thr:
                    ### For All Conditions ###
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
                    list_dir = [preds_dir, labels_dir, desc_dir]
                    split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

                    preds_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'preds')
                    labels_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'gts')
                    desc_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'desc')
                    split_path_road =path_dir + f'/{conf_thr}/' + road_cond_tag + '/val.txt'

                    preds_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'preds')
                    labels_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'gts')
                    desc_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'desc')
                    split_path_time = path_dir + f'/{conf_thr}/' + time_cond_tag + '/val.txt'

                    preds_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'preds')
                    labels_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'gts')
                    desc_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'desc')
                    split_path_weather =path_dir + f'/{conf_thr}/' + weather_cond_tag + '/val.txt'

                    os.makedirs(labels_dir_road, exist_ok=True)
                    os.makedirs(labels_dir_time, exist_ok=True)
                    os.makedirs(labels_dir_weather, exist_ok=True)
                    os.makedirs(desc_dir_road, exist_ok=True)
                    os.makedirs(desc_dir_time, exist_ok=True)
                    os.makedirs(desc_dir_weather, exist_ok=True)
                    os.makedirs(preds_dir_road, exist_ok=True)
                    os.makedirs(preds_dir_time, exist_ok=True)
                    os.makedirs(preds_dir_weather, exist_ok=True)

                    dict_out_current = self.network.list_modules[-1].get_pred_boxes_nms_for_single_datum(dict_out, conf_thr)

                    if dict_out_current is None:
                        continue

                    dict_out_current = dict_datum_to_kitti(self, dict_out_current)

                    if len(dict_out_current['kitti_labels']) == 0: # not eval emptry label
                        # with open(labels_dir + '/' + idx_name + '.txt', mode) as f:
                        #     f.write('\n')
                        pass
                    else:

                        ### Process Labels ###
                        for idx_label, label in enumerate(dict_out_current['kitti_labels']):
                            if idx_label == 0:
                                mode = 'w'
                            else:
                                mode = 'a'

                            with open(labels_dir + '/' + idx_name + '.txt', mode) as f:
                                f.write(label+'\n')
                            with open(labels_dir_road + '/' + idx_name + '.txt', mode) as f:
                                f.write(label+'\n')
                            with open(labels_dir_time + '/' + idx_name + '.txt', mode) as f:
                                f.write(label+'\n')
                            with open(labels_dir_weather + '/' + idx_name + '.txt', mode) as f:
                                f.write(label+'\n')

                        ### Process description ###
                        with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                            f.write(dict_out_current['kitti_desc'])
                        with open(desc_dir_road + '/' + idx_name + '.txt', 'w') as f:
                            f.write(dict_out_current['kitti_desc'])
                        with open(desc_dir_time + '/' + idx_name + '.txt', 'w') as f:
                            f.write(dict_out_current['kitti_desc'])
                        with open(desc_dir_weather + '/' + idx_name + '.txt', 'w') as f:
                            f.write(dict_out_current['kitti_desc'])

                        ### Process description ###
                        if len(dict_out_current['kitti_preds']) == 0:
                            with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                                f.write('\n')
                            with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                                f.write('\n')
                            with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                                f.write('\n')
                            with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                                f.write('\n')
                        else:
                            for idx_pred, pred in enumerate(dict_out_current['kitti_preds']):
                                if idx_pred == 0:
                                    mode = 'w'
                                else:
                                    mode = 'a'

                                with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                                    f.write(pred+'\n')
                                with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                                    f.write(pred+'\n')
                                with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                                    f.write(pred+'\n')
                                with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                                    f.write(pred+'\n')

                        str_log = idx_name + '\n'
                        with open(split_path, 'a') as f:
                            f.write(str_log)
                        with open(split_path_road, 'a') as f:
                            f.write(str_log)
                        with open(split_path_time, 'a') as f:
                            f.write(str_log)
                        with open(split_path_weather, 'a') as f:
                            f.write(str_log)
                tqdm_bar.update(1)
            tqdm_bar.close()

            ### Validate per conf ###
            all_condition_list = ['all'] + road_cond_list + time_cond_list + weather_cond_list
            for conf_thr in list_conf_thr:
                for condition in all_condition_list:
                    try:
                        preds_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'preds')
                        labels_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'gts')
                        desc_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'desc')
                        split_path = path_dir + f'/{conf_thr}/' + condition + '/val.txt'

                        dt_annos = kitti.get_label_annos(preds_dir)
                        val_ids = read_imageset_file(split_path)
                        gt_annos = kitti.get_label_annos(labels_dir, val_ids)
                        if self.val_iou_mode == 'all':
                            list_metrics = []
                            list_results = []
                            for idx_cls_val in self.list_care_cls_idx:
                                try:
                                    dict_metrics, result = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                                except Exception as e:
                                    print(e)

                                    dic_cls_val = self.cfg.VAL.DIC_CLASS_VAL
                                    cls_name = dic_cls_val[list(dic_cls_val.keys())[idx_cls_val]]
                                    dict_metrics = {
                                        'cls': cls_name,
                                        'iou': self.cfg.VAL.LIST_VAL_IOU,
                                        'bbox': [0., 0., 0.],
                                        'bev': [0., 0., 0.],
                                        '3d': [0., 0., 0.],
                                    }
                                    result = f'error occurs for {cls_name}\n'
                                list_metrics.append(dict_metrics)
                                list_results.append(result)

                            for dict_metrics, result in zip(list_metrics, list_results):
                                cls_name = dict_metrics['cls']
                                ious = dict_metrics['iou']
                                bevs = dict_metrics['bev']
                                ap3ds = dict_metrics['3d']
                                self.log_val.add_scalars(f'{log_header}/bev_conf_{conf_thr}', {
                                    f'iou_{ious[0]}_{cls_name}': bevs[0],
                                    f'iou_{ious[1]}_{cls_name}': bevs[1],
                                    f'iou_{ious[2]}_{cls_name}': bevs[2],
                                }, epoch)
                                self.log_val.add_scalars(f'{log_header}/3d_conf_{conf_thr}', {
                                    f'iou_{ious[0]}_{cls_name}': ap3ds[0],
                                    f'iou_{ious[1]}_{cls_name}': ap3ds[1],
                                    f'iou_{ious[2]}_{cls_name}': ap3ds[2],
                                }, epoch)
                                self.log_val.add_text(f'{log_header}/conf_{conf_thr}_{cls_name}', result, epoch)
                        elif self.val_iou_mode == 'each':

                            list_iou_mode = ['hard', 'mod', 'easy']
                            for idx_cls_val in self.list_care_cls_idx:
                                log_result = ''
                                for idx_mode, iou_mode in enumerate(list_iou_mode):
                                    try:
                                        dict_metrics, result = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, \
                                                                                            iou_mode=iou_mode, is_return_with_dict=True)
                                    except Exception as e:
                                        print(e)
                                        dic_cls_val = self.cfg.VAL.DIC_CLASS_VAL
                                        cls_name = dic_cls_val[list(dic_cls_val.keys())[idx_cls_val]]
                                        dict_metrics = {
                                            'cls': cls_name,
                                            'iou': [self.cfg.VAL.LIST_VAL_IOU[idx_mode]],
                                            'bbox': [0.],
                                            'bev': [0.],
                                            '3d': [0.],
                                        }
                                        result = f'error occurs for {cls_name} in iou {iou_mode}...\n'

                                    cls_name = dict_metrics['cls']
                                    ious = dict_metrics['iou']
                                    bevs = dict_metrics['bev']
                                    ap3ds = dict_metrics['3d']
                                    
                                    self.log_val.add_scalars(f'{log_header}/bev_conf_{conf_thr}', {
                                        f'{cls_name}_iou_{ious[0]}': bevs[0]
                                    }, epoch)
                                    self.log_val.add_scalars(f'{log_header}/3d_conf_{conf_thr}', {
                                        f'{cls_name}_iou_{ious[0]}': ap3ds[0]
                                    }, epoch)
                                    log_result += result
                                self.log_val.add_text(f'{log_header}/conf_{conf_thr}_{cls_name}', log_result, epoch)

                        print('Evaluation Condition: ', condition)

                        with open(os.path.join(path_dir, f'{conf_thr}', 'complete_results.txt'), 'a') as f:
                            for dic_metric in list_metrics:
                                print('='*25, '\n')
                                print('Cls: ', dic_metric['cls'])
                                print('IoU:', dic_metric['iou'])
                                print('BEV: ', dic_metric['bev'])
                                print('3D: ', dic_metric['3d'])
                                
                                f.write('Thres - ' + str(conf_thr) +  '- Condition - ' + condition + '\n')
                                f.write('cls: ' + dic_metric['cls'] + '\n')
                                f.write('iou: ')
                                for iou in dic_metric['iou']:
                                    f.write(str(iou) + ' ')
                                f.write('\n')
                                f.write('bev: ')
                                for bev in dic_metric['bev']:
                                    f.write(str(bev) + ' ')
                                f.write('\n')
                                f.write('3d  :')
                                for det3d in dic_metric['3d']:
                                    f.write(str(det3d) + ' ')
                                f.write('\n')
                    
                    except Exception as e:
                        print(e)
                        print('Condition Samples Not Found')

