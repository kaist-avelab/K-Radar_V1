"""
# -*- coding: utf-8 -*-
--------------------------------------------------------------------------------
# author: Donghee Paek, AVELab, KAIST
# date:   2021.12.28
# e-mail: donghee.paek@kaist.ac.kr
--------------------------------------------------------------------------------
# description: utils for common
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import time

from models.skeletons import build_skeleton
import datasets
from configs.config_general import IS_UBUNTU
import configs.config_general as cnf
from utils.util_geometry import *

__all__ = [ 'build_network', \
            'build_optimizer', \
            'build_dataset', \
            'build_scheduler', \
            'vis_tesseract_pline', \
            'set_random_seed', \
            'vis_tesseract_ra_bbox_pline', \
            'get_local_time_str', \
            'dict_datum_to_kitti', \
            'read_imageset_file', \
            ]

def build_network(p_pline):
    return build_skeleton(p_pline.cfg)

def build_optimizer(p_pline, model):
    lr = p_pline.cfg.OPTIMIZER.LR
    betas = p_pline.cfg.OPTIMIZER.BETAS
    weight_decay = p_pline.cfg.OPTIMIZER.WEIGHT_DECAY
    momentum = p_pline.cfg.OPTIMIZER.MOMENTUM
    
    params = model.parameters()
    if p_pline.cfg.OPTIMIZER.NAME == 'Adam':
        optimizer = torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif p_pline.cfg.OPTIMIZER.NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif p_pline.cfg.OPTIMIZER.NAME == 'SGD':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    return optimizer

def build_dataset(p_pline, split='train'):
    return datasets.__all__[p_pline.cfg.DATASET.NAME](cfg = p_pline.cfg, split=split)

def build_scheduler(p_pline, optimizer):
    max_epoch = p_pline.cfg.OPTIMIZER.MAX_EPOCH
    batch_size = p_pline.cfg.OPTIMIZER.BATCH_SIZE
    type_total_iter = p_pline.cfg.OPTIMIZER.TYPE_TOTAL_ITER
    try:
        min_lr = p_pline.cfg.OPTIMIZER.MIN_LR
    except:
        print('No Min LR in Config')
        min_lr = 0
    # print(p_pline.cfg.DATASET.NUM)
    if type_total_iter == 'every':
        total_iter = p_pline.cfg.DATASET.NUM // batch_size
    else:
        total_iter = (p_pline.cfg.DATASET.NUM // batch_size) * max_epoch
    if p_pline.cfg.OPTIMIZER.SCHEDULER is None:
        return None
    elif p_pline.cfg.OPTIMIZER.SCHEDULER == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iter, eta_min = min_lr)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def vis_tesseract_pline(p_pline, idx=0, vis_type='ra', is_in_deg=True, is_vis_local_maxima_along_range=False):
    '''
    * args
    *   idx: index of data
    *   vis_type: 'ra', 're', 'ae', 'all'
    '''
    datum = p_pline.dataset[idx]
    
    # mean doppler
    tesseract = datum['tesseract'].copy()
    tes_rae = np.mean(tesseract, axis=0)

    # visualize in color heatmap with 2D bbox (XY)
    tes_ra = np.mean(tes_rae, axis=2)
    tes_re = np.mean(tes_rae, axis=1)
    tes_ae = np.mean(tes_rae, axis=0)

    arr_range = p_pline.dataset.arr_range
    arr_azimuth = p_pline.dataset.arr_azimuth
    arr_elevation = p_pline.dataset.arr_elevation

    # print(tes_ra.shape)
    # print(tes_re.shape)
    # print(tes_ae.shape)
    # print(arr_range.shape)
    # print(arr_azimuth.shape)
    # print(arr_elevation.shape)
    
    ### Visualization ###
    if is_in_deg:
        arr_azimuth = arr_azimuth*180./np.pi
        arr_elevation = arr_elevation*180./np.pi

    if not IS_UBUNTU:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    if vis_type == 'ra':
        arr_0, arr_1 = np.meshgrid(arr_azimuth, arr_range)
        plt.pcolormesh(arr_0, arr_1, 10*np.log10(tes_ra), cmap='jet')

        plt.colorbar()
        plt.show()
    elif vis_type == 're':
        arr_0, arr_1 = np.meshgrid(arr_elevation, arr_range)

        tes_re_log_scale = 10*np.log10(tes_re)
        if is_vis_local_maxima_along_range:
            min_tes_re_log_scale = np.min(tes_re_log_scale)
            tes_re_local_maxima = np.ones_like(tes_re_log_scale)*min_tes_re_log_scale
            n_row, _ = tes_re_log_scale.shape
            for j in range(n_row):
                arg_maxima = np.argmax(tes_re_log_scale[j,:])
                tes_re_local_maxima[j, arg_maxima] = tes_re_log_scale[j, arg_maxima]
            plt.pcolormesh(arr_0, arr_1, tes_re_local_maxima, cmap='jet')
        else:
            plt.pcolormesh(arr_0, arr_1, tes_re_log_scale, cmap='jet')
        
        plt.colorbar()
        plt.show()
    elif vis_type == 'ae':
        arr_0, arr_1 = np.meshgrid(arr_elevation, arr_azimuth)
        plt.pcolormesh(arr_0, arr_1, 10*np.log10(tes_ae), cmap='jet')
        plt.colorbar()
        plt.show()
    elif vis_type == 'all':
        # fig, ax = plt.subplots(1, 1)
        # ax.pcolormesh(arr_0, arr_1, 10*np.log10(tes_ra), cmap='jet')

        return

def vis_tesseract_ra_bbox_pline(p_pline, idx, roi_x, roi_y, is_with_label=True, is_in_deg=True):
    datum = p_pline.dataset[idx]
    
    # mean doppler
    tesseract = datum['tesseract'].copy()
    tes_rae = np.mean(tesseract, axis=0)
    tes_ra = np.mean(tes_rae, axis=2)

    arr_range = p_pline.dataset.arr_range
    arr_azimuth = p_pline.dataset.arr_azimuth
    if is_in_deg:
        arr_azimuth = arr_azimuth*180./np.pi

    if not IS_UBUNTU:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    arr_0, arr_1 = np.meshgrid(arr_azimuth, arr_range)

    # 이미지 공백 제거
    # https://frhyme.github.io/python-lib/img_savefig_%EA%B3%B5%EB%B0%B1%EC%A0%9C%EA%B1%B0/
    height, width = np.shape(tes_ra)
    # print(height, width)
    figsize = (1, height/width) if height>=width else (width/height, 1)
    plt.figure(figsize=figsize)
    plt.pcolormesh(arr_0, arr_1, 10*np.log10(tes_ra), cmap='jet')
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
    plt.savefig('./resources/imgs/img_tes_ra.png', bbox_inces='tight', pad_inches=0, dpi=300)
    
    # match the size of image
    temp_img = cv2.imread('./resources/imgs/img_tes_ra.png')
    temp_row, temp_col, _ = temp_img.shape
    # print(temp_row, temp_col)
    if not (temp_row == height and temp_col == width):
        temp_img_new = cv2.resize(temp_img, (width, height), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./resources/imgs/img_tes_ra.png', temp_img_new)

    plt.close()
    plt.pcolormesh(arr_0, arr_1, 10*np.log10(tes_ra), cmap='jet')
    plt.colorbar()
    plt.savefig('./resources/imgs/plot_tes_ra.png', dpi=300)

    # Polar to Cartesian (Should flip image)
    ra = cv2.imread('./resources/imgs/img_tes_ra.png')
    ra = np.flip(ra, axis=0)

    # This code is why we have to flip
    # ra_gray = cv2.cvtColor(ra, cv2.COLOR_BGR2GRAY)
    # plt.figure()
    # plt.pcolormesh(arr_0, arr_1, ra_gray.astype(float))
    # plt.show()
    # This code is why we have to flip

    arr_yx, arr_y, arr_x  = get_xy_from_ra_color(ra, \
        arr_range, arr_azimuth, roi_x=roi_x, roi_y=roi_y, is_in_deg=True)
    
    if is_with_label:
        label = datum['meta']['labels']
        # print(datum['meta'])
        arr_yx_bbox = draw_labels_in_yx_bgr(arr_yx, arr_y, arr_x, label)

    # flip before show
    arr_yx = arr_yx.transpose((1,0,2))
    arr_yx = np.flip(arr_yx, axis=(0,1))
    
    arr_yx_bbox = arr_yx_bbox.transpose((1,0,2))
    arr_yx_bbox = np.flip(arr_yx_bbox, axis=(0,1))
    # flip before show

    cv2.imshow('Cartesian', arr_yx)
    cv2.imshow('Cartesian (bbox)', arr_yx_bbox)
    cv2.imshow('Front image', cv2.imread(datum['meta']['path_img']))
    plt.show()

    
def draw_labels_in_yx_bgr(arr_yx_in, arr_y_in, arr_x_in, label_in, is_with_bbox_mask=True):
    arr_yx = arr_yx_in.copy()
    arr_y = arr_y_in.copy()
    arr_x = arr_x_in.copy()
    label = label_in.copy()

    y_m_per_pix = np.mean(arr_y[1:] - arr_y[:-1])
    x_m_per_pix = np.mean(arr_x[1:] - arr_x[:-1])
    # print(y_m_per_pix, x_m_per_pix)

    y_min = np.min(arr_y)
    x_min = np.min(arr_x)

    if is_with_bbox_mask:
        row, col, _ = arr_yx.shape
        arr_yx_mask = np.zeros((row, col), dtype=float) # 1 or 0

    dic_cls_bgr = cnf.DIC_CLS_BGR

    for obj in label:
        cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj = obj

        # calib is already reflected in dataset
        color = dic_cls_bgr[cls_name]
        
        x_pix = (x-x_min)/x_m_per_pix
        y_pix = (y-y_min)/y_m_per_pix
        # y_pix = (-y-y_min)/y_m_per_pix
        # Why? Row 좌표계!, Theta도 Checking 필요
        
        # y_pix = (y-y_min)/y_m_per_pix
        l_pix = l/x_m_per_pix
        w_pix = w/y_m_per_pix
        
        pts = [ [l_pix/2, w_pix/2],
                [l_pix/2, -w_pix/2],
                [-l_pix/2, -w_pix/2],
                [-l_pix/2, w_pix/2]]
        # front left, front right, back right, back left

        # rotation
        # print(f'th = {theta}')
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        # rotation & translation
        pts = list(map(lambda pt: [ x_pix +cos_th*pt[0]-sin_th*pt[1], \
                                    y_pix +sin_th*pt[0]+cos_th*pt[1] ], pts))
        pt_front = (int(np.around((pts[0][0]+pts[1][0])/2)), int(np.around((pts[0][1]+pts[1][1])/2)))
        
        # to integer and tuple
        pts = list(map(lambda pt: (int(np.around(pt[0])), int(np.around(pt[1]))), pts))

        arr_yx = cv2.line(arr_yx, pts[0], pts[1], color, 1)
        arr_yx = cv2.line(arr_yx, pts[1], pts[2], color, 1)
        arr_yx = cv2.line(arr_yx, pts[2], pts[3], color, 1)
        arr_yx = cv2.line(arr_yx, pts[3], pts[0], color, 1)

        # front and center
        pt_cen = (int(np.around(x_pix)), int(np.around(y_pix)))
        arr_yx = cv2.line(arr_yx, pt_cen, pt_front, color, 1)

        # make center black
        arr_yx = cv2.circle(arr_yx, pt_cen, 1, (0,0,0), thickness=-1)

    return arr_yx

def get_local_time_str():
    now = time.localtime()
    return f'{now.tm_mon}_{now.tm_mday}_{now.tm_hour}_{now.tm_min}_{now.tm_sec}'

def dict_datum_to_kitti(p_pline, data_dic):
    '''
        assuming batch size = 1
    '''
    kitti_preds = []
    kitti_labels = []

    if 'pred_boxes_nms' in data_dic.keys():
        for pred_idx, pred_box in enumerate(data_dic['pred_boxes_nms']):
            # print(pred_box.shape)
            score, xc, yc, zc, xl, yl, zl, rot = pred_box
            # We need class id
            # KITTI Example: Car -1 -1 -4.2780 668.7884 173.1068 727.8801 198.9699 1.4607 1.7795 4.5159 5.3105 1.4764 43.1853 -4.1569 0.9903
            cls_idx = data_dic['pred_cls_ids'][pred_idx].item()
            cls_id = p_pline.val_cls_pred[cls_idx] # 'Car' # just change cls id
            # print(cls_id)
            header = '-1 -1 0 50 50 150 150'
            box_centers = str(yc.item()) + ' ' + str(zc.item()) + ' ' + str(xc.item())
            box_dim = str(zl.item()) + ' ' + str(yl.item()) + ' ' + str(xl.item())
            str_rot = str(rot.item())
            str_score = str(score.item())
            kitti_pred = cls_id + ' ' + header  + ' ' + box_dim + ' ' + box_centers + ' ' + str_rot + ' ' + str_score
            kitti_preds.append(kitti_pred)

    if len(kitti_preds) == 0:
        kitti_dummy = 'dummy -1 -1 0 0 0 0 0 0 0 0 0 0 0 0 0'
        kitti_preds.append(kitti_dummy)
    # print('='*50)
    # print('labels')

    for batch_id, batch_labels in enumerate(data_dic['labels']):
        # print(batch_id) # check only 0
        for label_id, label in enumerate(batch_labels):
            cls_name, cls_idx, (xc, yc, zc, rz, xl, yl, zl), _ = label
            xc, yc, zc, rz, xl, yl, zl = np.round(xc, 2), np.round(yc, 2), np.round(zc, 2), np.round(rz, 2), np.round(xl, 2), np.round(yl, 2), np.round(zl, 2),
            # print(cls_name)
            cls_id = p_pline.val_cls_label[cls_idx]
            # print(cls_id)
            header = '0.00 0 0 50 50 150 150'
            box_centers = str(yc) + ' ' + str(zc) + ' ' + str(xc) # xcam, ycam, zcam
            box_dim = str(zl) + ' ' + str(yl) + ' ' + str(xl) # height(ycaml), width(xcaml), length(zcaml)
            str_rot = str(rz)

            kitti_label = cls_id + ' ' + header  + ' ' + box_dim  + ' ' + box_centers + ' ' + str_rot
            kitti_labels.append(kitti_label)
    
    # print('-'*50)

    desc_dic = data_dic['pred_desc']
    capture_time = desc_dic['capture_time']
    road_type = desc_dic['road_type']
    climate = desc_dic['climate']
    is_nms = 'nms_done' if desc_dic['is_nms'] else 'nms_error'

    data_dic['kitti_preds'] = kitti_preds
    data_dic['kitti_labels'] = kitti_labels
    data_dic['kitti_desc'] = f'{capture_time}\n{road_type}\n{climate}\n{is_nms}'
    

    return data_dic

def read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]
