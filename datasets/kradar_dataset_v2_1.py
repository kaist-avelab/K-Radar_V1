'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

from fileinput import filename
import os
from random import sample
import numpy as np
import torch
from torch.utils.data import Dataset
import sys

import os.path as osp
from glob import glob
from scipy.io import loadmat # from matlab
import pickle
import time

try:
    from utils.util_geometry import *
    from utils.util_geometry import Object3D
    from utils.util_dataset import *
except:
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    from utils.util_geometry import *
    from utils.util_geometry import Object3D
    from utils.util_dataset import *

class KRadarDataset_v2_1(Dataset):
    def __init__(self, cfg=None, split='train'):
        super().__init__()
        self.cfg = cfg

        ### Load label paths wrt split ###
        # load label paths
        self.split = split # 'train', 'test'
        self.dict_split = self.get_split_dict(self.cfg.DATASET.SPLIT.PATH_SPLIT[split])
        self.label_paths = [] # a list of dic
        for dir_seq in self.cfg.DATASET.SPLIT.LIST_DIR:
            list_seq = os.listdir(dir_seq)
            for seq in list_seq:
                seq_label_paths = sorted(glob(osp.join(dir_seq, seq, 'info_label', '*.txt')))
                seq_label_paths = list(filter(lambda x: (x.split('/')[-1].split('.')[0] in self.dict_split[seq]), seq_label_paths))
                self.label_paths.extend(seq_label_paths)

        # load generated labels (Gaussian confidence)
        self.is_use_gen_labels = self.cfg.DATASET.LABEL.IS_USE_PREDEFINED_LABEL
        if self.is_use_gen_labels:
            self.pre_label_dir = self.cfg.DATASET.LABEL.PRE_LABEL_DIR
        ### Load label paths wrt split ###

        ### Selecting Radar/Lidar/Camera (Unification of coordinate systems) ###
        self.type_coord = self.cfg.DATASET.MODALITY.TYPE_COORD # 1: Radar, 2: Lidar, 3: Camera (TBD)
        self.is_consider_rdr = self.cfg.DATASET.MODALITY.IS_CONSIDER_RDR
        self.is_consider_ldr = self.cfg.DATASET.MODALITY.IS_CONSIDER_LDR
        ### Selecting Radar/Lidar/Camera (Unification of coordinate systems) ###

        ### Radar Tesseract ###
        if self.cfg.DATASET.GET_ITEM['rdr_tesseract']:
            # load physical values
            self.arr_range, self.arr_azimuth, self.arr_elevation = self.load_physical_values()
            # consider roi
            self.is_consider_roi_rdr = cfg.DATASET.RDR.IS_CONSIDER_ROI_RDR
            if self.is_consider_roi_rdr:
                self.consider_roi_rdr(cfg.DATASET.RDR.RDR_POLAR_ROI)
        ### Radar Tesseract ###

        ### Radar Cube ###
        self.is_get_cube_dop = False
        if self.cfg.DATASET.GET_ITEM['rdr_cube']:
            # dealing cube data
            _, _, _, self.arr_doppler = self.load_physical_values(is_with_doppler=True)
            self.is_count_minus_1_for_bev = cfg.DATASET.RDR_CUBE.IS_COUNT_MINUS_ONE_FOR_BEV # To make BEV -> averaging power
            self.arr_bev_none_minus_1 = None
            self.arr_z_cb = np.arange(-30, 30, 0.4)
            self.arr_y_cb = np.arange(-80, 80, 0.4)
            self.arr_x_cb = np.arange(0, 100, 0.4)
            self.is_consider_roi_rdr_cb = cfg.DATASET.RDR_CUBE.IS_CONSIDER_ROI_RDR_CB
            if self.is_consider_roi_rdr_cb:
                self.consider_roi_cube(cfg.DATASET.RDR_CUBE.RDR_CB_ROI)
                if cfg.DATASET.RDR_CUBE.CONSIDER_ROI_ORDER == 'cube -> num':
                    self.consider_roi_order = 1
                elif cfg.DATASET.RDR_CUBE.CONSIDER_ROI_ORDER == 'num -> cube':
                    self.consider_roi_order = 2
                else:
                    raise AttributeError('Check consider roi order in cfg')
                if cfg.DATASET.RDR_CUBE.BEV_DIVIDE_WITH == 'bin_z':
                    self.bev_divide_with = 1
                elif cfg.DATASET.RDR_CUBE.BEV_DIVIDE_WITH == 'none_minus_1':
                    self.bev_divide_with = 2
                else:
                    raise AttributeError('Check consider bev divide with in cfg')
            try:
                self.is_get_cube_dop = cfg.DATASET.RDR_CUBE.DOPPLER.IS_GET_DOPPLER
                self.is_dop_another_dir = cfg.DATASET.RDR_CUBE.DOPPLER.IS_ANOTHER_DIR
                self.dir_dop = cfg.DATASET.RDR_CUBE.DOPPLER.DIR_DOPPLER
            except:
                print('not using doppler cube info')
                self.is_get_cube_dop = False
                self.is_dop_another_dir = False
                self.dir_dop = None
        ### Radar Cube ###

        ### Considering Label ###
        if self.cfg.DATASET.LABEL.ROI_CONSIDER_LABEL_TYPE == 'cube':
            x_roi, y_roi, z_roi = cfg.DATASET.RDR_CUBE.RDR_CB_ROI['x'], \
                cfg.DATASET.RDR_CUBE.RDR_CB_ROI['y'], cfg.DATASET.RDR_CUBE.RDR_CB_ROI['z']
            x_min, x_max = [0, 150] if x_roi is None else x_roi
            y_min, y_max = [-160, 160] if y_roi is None else y_roi
            z_min, z_max = [-150, 150] if z_roi is None else z_roi
            self.roi_label = [x_min, y_min, z_min, x_max, y_max, z_max]
            # print(self.roi_label)
            self.is_roi_check_with_azimuth = self.cfg.DATASET.LABEL.IS_CHECK_VALID_WITH_AZIMUTH
            self.max_azimtuth_rad = self.cfg.DATASET.LABEL.MAX_AZIMUTH_DEGREE
            self.max_azimtuth_rad = [self.max_azimtuth_rad[0]*np.pi/180., self.max_azimtuth_rad[1]*np.pi/180.]
            # print(self.max_azimtuth_rad)
        elif self.cfg.DATASET.LABEL.ROI_CONSIDER_LABEL_TYPE == 'default':
            self.roi_label = self.cfg.DATASET.LABEL.ROI_CONSIDER_LABEL # only tesseract
            self.is_roi_check_with_azimuth = False
        else:
            raise AttributeError('ROI_CONSIDER_LABEL_TYPE should be cube or default.')
        ### Considering Label ###
        
        ### Lidar ###
        # (TBD)
        ### Lidar ###

        ### Camera ###
        # (TBD)
        ### Camera ###

    def get_split_dict(self, path_split):
        f = open(path_split, 'r')
        lines = f.readlines()
        f.close

        dict_seq = dict()
        for line in lines:
            seq = line.split(',')[0]
            label = line.split(',')[1].split('.')[0]

            if not (seq in list(dict_seq.keys())):
                dict_seq[seq] = []
            
            dict_seq[seq].append(label)

        return dict_seq

    def load_physical_values(self, is_in_rad=True, is_with_doppler=False):
        temp_values = loadmat('./resources/info_arr.mat')
        arr_range = temp_values['arrRange']
        if is_in_rad:
            deg2rad = np.pi/180.
            arr_azimuth = temp_values['arrAzimuth']*deg2rad
            arr_elevation = temp_values['arrElevation']*deg2rad
        else:
            arr_azimuth = temp_values['arrAzimuth']
            arr_elevation = temp_values['arrElevation']
        _, num_0 = arr_range.shape
        _, num_1 = arr_azimuth.shape
        _, num_2 = arr_elevation.shape
        arr_range = arr_range.reshape((num_0,))
        arr_azimuth = arr_azimuth.reshape((num_1,))
        arr_elevation = arr_elevation.reshape((num_2,))
        if is_with_doppler:
            arr_doppler = loadmat('./resources/arr_doppler.mat')['arr_doppler']
            _, num_3 = arr_doppler.shape
            arr_doppler = arr_doppler.reshape((num_3,))
            return arr_range, arr_azimuth, arr_elevation, arr_doppler
        else:
            return arr_range, arr_azimuth, arr_elevation

    def consider_roi_rdr(self, roi_polar, is_reflect_to_cfg=True):
        self.list_roi_idx = [0, len(self.arr_range)-1, \
            0, len(self.arr_azimuth)-1, 0, len(self.arr_elevation)-1]

        idx_attr = 0
        deg2rad = np.pi/180.
        rad2deg = 180./np.pi

        for k, v in roi_polar.items():
            if v is not None:
                min_max = (np.array(v)*deg2rad).tolist() if idx_attr > 0 else v
                arr_roi, idx_min, idx_max = self.get_arr_in_roi(getattr(self, f'arr_{k}'), min_max)
                setattr(self, f'arr_{k}', arr_roi)
                self.list_roi_idx[idx_attr*2] = idx_min
                self.list_roi_idx[idx_attr*2+1] = idx_max
                
                if is_reflect_to_cfg:
                    v_new = [arr_roi[0], arr_roi[-1]]
                    v_new =  (np.array(v_new)*rad2deg) if idx_attr > 0 else v_new
                    self.cfg.DATASET.RDR.RDR_POLAR_ROI[k] = v_new
            idx_attr += 1

        ### This is for checking ###
        # print(self.cfg.DATASET.RDR.RDR_POLAR_ROI)
        # print(self.arr_range)
        # print(self.arr_azimuth*rad2deg)
        ### This is for checking ###

    def consider_roi_cube(self, roi_cart, is_reflect_to_cfg=True):
        self.list_roi_idx_cb = [0, len(self.arr_z_cb)-1, \
            0, len(self.arr_y_cb)-1, 0, len(self.arr_x_cb)-1]

        idx_attr = 0
        for k, v in roi_cart.items():
            if v is not None:
                min_max = np.array(v).tolist()
                # print(min_max)
                arr_roi, idx_min, idx_max = self.get_arr_in_roi(getattr(self, f'arr_{k}_cb'), min_max)
                setattr(self, f'arr_{k}_cb', arr_roi)
                self.list_roi_idx_cb[idx_attr*2] = idx_min
                self.list_roi_idx_cb[idx_attr*2+1] = idx_max

                if is_reflect_to_cfg:
                    v_new = [arr_roi[0], arr_roi[-1]]
                    v_new = np.array(v_new)
                    self.cfg.DATASET.RDR_CUBE.RDR_CB_ROI[k] = v_new
            idx_attr += 1

    def get_arr_in_roi(self, arr, min_max):
        min_val, max_val = min_max
        idx_min = np.argmin(abs(arr-min_val))
        idx_max = np.argmin(abs(arr-max_val))
        
        return arr[idx_min:idx_max+1], idx_min, idx_max

    def get_calib_info(self, path_calib, is_z_offset_from_cfg=True):
        '''
        * return: [X, Y, Z]
        * if you want to get frame difference, get list_calib[0]
        '''
        with open(path_calib) as f:
            lines = f.readlines()
            f.close()
            
        try:
            list_calib = list(map(lambda x: float(x), lines[1].split(',')))
            # list_calib[0] # frame difference
            list_values = [list_calib[1], list_calib[2]] # X, Y
            
            if is_z_offset_from_cfg:
                list_values.append(self.cfg.DATASET.Z_OFFSET) # Z
            else:
                list_values.append(0.)

            return np.array(list_values)
        except:
            raise FileNotFoundError('no calib info')

    def get_tuple_object(self, line, calib_info, is_heading_in_rad=True):
        '''
        * in : e.g., '*, 0, Sedan, 3.8, 5.535, -1.0, -93.1155, 2.112, 1.0347, 0.95' --> One Example
        * in : e.g., '*, 0, 0, Sedan, 3.8, 5.535, -1.0, -93.1155, 2.112, 1.0347, 0.95' --> There are labels like this too
        * out: tuple ('Sedan', idx_cls, [x, y, z, theta, l, w, h], idx_obj)
        *       None if idx_cls == -1 or header != '*'
        '''
        list_values = line.split(',')

        if list_values[0] != '*':
            return None

        offset = 0
        if(len(list_values)) == 11:
            offset = 1
        cls_name = list_values[2+offset][1:]

        idx_cls = self.cfg.DATASET.CLASS_ID[cls_name]

        if idx_cls == -1: # Considering None as -1
            return None

        idx_obj = int(list_values[1+offset])
        x = float(list_values[3+offset])
        y = float(list_values[4+offset])
        z = float(list_values[5+offset])
        theta = float(list_values[6+offset])
        if is_heading_in_rad:
            theta = theta*np.pi/180.
        l = 2*float(list_values[7+offset])
        w = 2*float(list_values[8+offset])
        h = 2*float(list_values[9+offset])

        if self.type_coord == 1: # radar coord
            # print('calib_info = ', calib_info)
            x = x+calib_info[0]
            y = y+calib_info[1]
            z = z+calib_info[2]

        # if the label is in roi
        # print('* x, y, z: ', x, y, z)
        # print('* roi_label: ', self.roi_label)

        x_min, y_min, z_min, x_max, y_max, z_max = self.roi_label
        if ((x > x_min) and (x < x_max) and \
            (y > y_min) and (y < y_max) and \
            (z > z_min) and (z < z_max)):
            # print('* here 1')

            if self.is_roi_check_with_azimuth:
                min_azi, max_azi = self.max_azimtuth_rad
                # print('* min, max: ', min_azi, max_azi)
                obj3d = Object3D(x, y, z, l, w, h, theta)
                pts = [obj3d.corners[0,:], obj3d.corners[2,:], obj3d.corners[4,:], obj3d.corners[6,:]]
                for pt in pts:
                    azimuth_apex = np.arctan2(-pt[1], pt[0])
                    # print(azimuth_apex)
                    if (azimuth_apex < min_azi) or (azimuth_apex > max_azi):
                        # print('* here 2')
                        return None
            # print('* here 3')
            return (cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj)
        else:
            # print('* here 4')
            return None

    def get_label_bboxes(self, path_label, calib_info):
        with open(path_label, 'r') as f:
            lines = f.readlines()
            f.close()
        # print('* lines : ', lines)
        line_objects = lines[1:]
        # print('* line_objs: ', line_objects)
        list_objects = []

        # print(dict_temp['meta']['path_label'])
        for line in line_objects:
            temp_tuple = self.get_tuple_object(line, calib_info)
            if temp_tuple is not None:
                list_objects.append(temp_tuple)

        return list_objects

    def get_tesseract(self, path_tesseract, is_in_DRAE=True, is_in_3d=False, is_in_log=False):
        # Otherwise you make the input as 4D, you should not get the data as log scale
        arr_tesseract = loadmat(path_tesseract)['arrDREA']
        
        if is_in_DRAE:
            arr_tesseract = np.transpose(arr_tesseract, (0, 1, 3, 2))

        ### considering ROI ###
        if self.is_consider_roi_rdr:
            # print(self.list_roi_idx)
            idx_r_0, idx_r_1, idx_a_0, idx_a_1, \
                idx_e_0, idx_e_1 = self.list_roi_idx
            # Python slicing grammar (+1)
            arr_tesseract = arr_tesseract[:,idx_r_0:idx_r_1+1,\
                idx_a_0:idx_a_1+1,idx_e_0:idx_e_1+1]
        ### considering ROI ###

        # Dimension reduction -> log operation
        if is_in_3d:
            arr_tesseract = np.mean(arr_tesseract, axis=3) # reduce elevation

        if is_in_log:
            arr_tesseract = 10*np.log10(arr_tesseract)

        return arr_tesseract

    def get_cube(self, path_cube, is_in_log=False, mode=0):
        '''
        * mode 0: arr_cube, mask, cnt
        * mode 1: arr_cube
        '''
        arr_cube = np.flip(loadmat(path_cube)['arr_zyx'], axis=0) # z-axis is flipped

        # print(arr_cube.shape)
        # print(np.count_nonzero(arr_cube==-1.))

        if (self.is_consider_roi_rdr_cb) & (self.consider_roi_order == 1):
            idx_z_min, idx_z_max, idx_y_min, idx_y_max, idx_x_min, idx_x_max = self.list_roi_idx_cb
            arr_cube = arr_cube[idx_z_min:idx_z_max+1,idx_y_min:idx_y_max+1,idx_x_min:idx_x_max+1]
        
        # print(arr_cube.shape)
        
        if self.is_count_minus_1_for_bev:
            bin_z = len(self.arr_z_cb)
            if self.bev_divide_with == 1:
                bin_y = len(self.arr_y_cb)
                bin_x = len(self.arr_x_cb)
                # print(bin_z, bin_y, bin_x)
                arr_bev_none_minus_1 = np.full((bin_y, bin_x), bin_z)
            elif self.bev_divide_with == 2:
                arr_bev_none_minus_1 = bin_z-np.count_nonzero(arr_cube==-1., axis=0)
                arr_bev_none_minus_1 = np.maximum(arr_bev_none_minus_1, 1) # evade divide 0
            # print('* max: ', np.max(arr_bev_none_minus_1))
            # print('* min: ', np.min(arr_bev_none_minus_1))

        # print(arr_bev_none_minus_1.shape)

        if (self.is_consider_roi_rdr_cb) & (self.consider_roi_order == 2):
            idx_z_min, idx_z_max, idx_y_min, idx_y_max, idx_x_min, idx_x_max = self.list_roi_idx_cb
            # print(idx_z_min, idx_z_max)
            arr_cube = arr_cube[idx_z_min:idx_z_max+1,idx_y_min:idx_y_max+1,idx_x_min:idx_x_max+1]
            if self.is_count_minus_1_for_bev:
                arr_bev_none_minus_1 = arr_bev_none_minus_1[idx_y_min:idx_y_max+1, idx_x_min:idx_x_max+1]

        # print(arr_bev_none_minus_1.shape)

        if is_in_log:
            arr_cube[np.where(arr_cube==-1.)]= 1.
            # arr_cube = np.maximum(arr_cube, 1.) # get rid of -1 before log
            arr_cube = 10*np.log10(arr_cube)
        else:
            arr_cube = np.maximum(arr_cube, 0.)

        none_zero_mask = np.nonzero(arr_cube)

        # print(arr_cube.shape)

        if mode == 0:
            return arr_cube, none_zero_mask, arr_bev_none_minus_1
        elif mode == 1:
            return arr_cube

    def get_cube_doppler(self, path_cube_doppler, dummy_value=0.):
        arr_cube = np.flip(loadmat(path_cube_doppler)['arr_zyx'], axis=0)
        # print(np.count_nonzero(arr_cube==-1.)) # no value -1. in doppler cube

        ### Change -1. to -10. in server ###
        arr_cube[np.where(arr_cube==-10.)] = dummy_value

        if self.is_consider_roi_rdr_cb:
            idx_z_min, idx_z_max, idx_y_min, idx_y_max, idx_x_min, idx_x_max = self.list_roi_idx_cb
            arr_cube = arr_cube[idx_z_min:idx_z_max+1,idx_y_min:idx_y_max+1,idx_x_min:idx_x_max+1]

        arr_cube = arr_cube + 1.9326

        return arr_cube

    def get_pc_lidar(self, path_lidar, calib_info=None):
        pc_lidar = []
        with open(path_lidar, 'r') as f:
            lines = [line.rstrip('\n') for line in f][13:]
            pc_lidar = [point.split() for point in lines]
            f.close()
        pc_lidar = np.array(pc_lidar, dtype = float).reshape(-1, 9)[:, :4]

        ### Filter out missing values ###
        pc_lidar = pc_lidar[np.where(pc_lidar[:, 0] > 0.01)].reshape(-1, 4)
        ### Filter out missing values ###

        if self.type_coord == 1: # rdr
            if calib_info is None:
                raise AttributeError('Insert calib info!')
            else:
                pc_lidar = np.array(list(map(lambda x: \
                    [x[0]+calib_info[0], x[1]+calib_info[1], x[2]+calib_info[2], x[3]],\
                    pc_lidar.tolist())))

        return pc_lidar

    def get_gen_conf_label(self, path_label):
        # label dir, exp name, sequence, file name
        file_name = path_label.split('/')[-1].split('.')[0]
        path_gen = osp.join(self.pre_label_dir, path_label.split('/')[-3], f'{file_name}.bin')
        with open(path_gen, 'rb') as f:
            gen_conf_label = pickle.load(f)
            f.close()

        return np.nan_to_num(gen_conf_label, nan=0.0)

    '''
    * [ Visualization Functions ]
    * For efficiency, we pull out the vis code to util_dataset
    '''
    
    ### 4D DRAE Radar Tensor (i.e., Tesseract) ###
    # To use radar tensor functions, GET_ITEM['tesseract'] = True
    def show_radar_tensor_bev(self, dict_item, bboxes=None, \
            roi_x = [0, 0.4, 100], roi_y = [-50, 0.4, 50], is_return_bbox_bev_tensor=False):        
        return func_show_radar_tensor_bev(self, dict_item, roi_x, roi_y, is_return_bbox_bev_tensor)

    def show_lidar_point_cloud(self, dict_item, bboxes=None, \
            roi_x=[0, 100], roi_y=[-50, 50], roi_z=[-10, 10]):
        func_show_lidar_point_cloud(self, dict_item, bboxes, roi_x, roi_y, roi_z)

    def show_gaussian_confidence_cart(self, roi_x=[0, 0.2, 100], roi_y=[-80, 0.2, 80], bboxes=None):
        func_show_gaussian_confidence_cart(self, roi_x, roi_y, bboxes)

    def show_gaussian_confidence_polar_color(self, arr_range=None, arr_azimuth=None, \
            roi_x=[0, 0.2, 100], roi_y=[-80, 0.2, 80], bboxes=None):
        func_show_gaussian_confidence_polar_color(self, arr_range, arr_azimuth, roi_x, roi_y, bboxes)

    def show_gaussian_confidence_polar(self, arr_range=None, arr_azimuth=None, \
            roi_x=[0, 0.2, 100], roi_y=[-80, 0.2, 80], bboxes=None):
        func_show_gaussian_confidence_polar(self, arr_range, arr_azimuth, roi_x, roi_y, bboxes)

    def show_heatmap_polar_with_bbox(self, idx_datum, scale=4):
        func_show_heatmap_polar_with_bbox(self, idx_datum, scale)
    
    def generate_gaussian_conf_labels(self, dir_gen, gen_type='polar', \
            roi_x_res=[0.00, 0.16, 69.12], roi_y_res=[-39.68, 0.16, 39.68]):
        func_generate_gaussian_conf_labels(self, dir_gen, gen_type, roi_x_res, roi_y_res)

    def show_rdr_pc_tesseract(self, dict_item, bboxes=None, cfar_params = [25, 8, 0.01], \
            roi_x=[0, 100], roi_y=[-50, 50], roi_z=[-10, 10], is_with_lidar=True):
        func_show_rdr_pc_tesseract(self, dict_item, bboxes, cfar_params, roi_x, roi_y, roi_z, is_with_lidar)
    ### 4D DRAE Radar Tensor (i.e., Tesseract) ###

    ### Radar Cube ###
    def show_radar_cube_bev(self, dict_item, bboxes=None, magnifying=4, is_with_doppler = False, is_with_log = False):
        func_show_radar_cube_bev(self, dict_item, bboxes, magnifying, is_with_doppler, is_with_log)

    def show_sliced_radar_cube(self, dict_item, bboxes=None, magnifying=4, idx_custom_slice=None):
        func_show_sliced_radar_cube(self, dict_item, bboxes, magnifying, idx_datum, idx_custom_slice)
    
    def show_rdr_pc_cube(self, dict_item, bboxes=None, cfar_params = [25, 8, 0.01], axis='x', is_with_lidar=True):
        func_show_rdr_pc_cube(self, dict_item, bboxes, cfar_params, axis, is_with_lidar)
    ### Radar Cube ###

    def get_description(self, path_desc):
        try:
            f = open(path_desc)
            line = f.readline()
            road_type, capture_time, climate = line.split(',')
            dict_desc = {
                'capture_time': capture_time,
                'road_type': road_type,
                'climate': climate,
            }
            f.close()
        except:
            raise FileNotFoundError(f'* check {path_desc}')
        
        return dict_desc
        
    def __len__(self):
        return len(self.label_paths)

    def get_data_indices(self, label_path):
        f = open(label_path, 'r')
        line = f.readlines()[0]
        f.close()

        seq_id = label_path.split('/')[-3]

        ### OUT OF INDEX HERE WHEN IT IS NOT ###
        # print(label_path)
        # print(line)
        rdr_idx, ldr_idx, camf_idx, _, _ = line.split(',')[0].split('=')[1].split('_')

        return seq_id, rdr_idx, ldr_idx, camf_idx
    
    def __getitem__(self, idx):
        try:
            # t1 = time.time()
            path_label = self.label_paths[idx]
            # print(label_path)
            seq_id, radar_idx, lidar_idx, camf_idx = self.get_data_indices(path_label)

            ### Use this when self.get_data_indices does not work ###
            # seq_id = label_path.split('/')[-3]
            # radar_idx = label_path.split('/')[-1].split('_')[0]
            # lidar_idx = label_path.split('/')[-1].split('_')[1].split('.')[0]
            # camf_idx = str('00000')
            ### Use this when self.get_data_indices does not work ###
            
            ### Use this when generating pre-defined labels ###
            # if int(seq_id) < {HERE SHOULD BE SEQ ID}:
            #     return 0
            ### Use this when generating pre-defined labels ###

            path_header = path_label.split('/')[:-2]
            path_radar_tesseract = '/'+os.path.join(*path_header, 'radar_tesseract', 'tesseract_'+radar_idx+'.mat')
            path_radar_cube = '/'+os.path.join(*path_header, 'radar_zyx_cube', 'cube_'+radar_idx+'.mat')
            path_radar_bev_img = '/'+os.path.join(*path_header, 'radar_bev_image', 'radar_bev_100_'+radar_idx+'.png')
            path_lidar_bev_img = '/'+os.path.join(*path_header, 'lidar_bev_image', 'lidar_bev_100_'+lidar_idx+'.png')
            path_lidar_pc_64 = '/'+os.path.join(*path_header, 'os2-64', 'os2-64_'+lidar_idx+'.pcd')
            path_lidar_pc_128 = '/'+os.path.join(*path_header, 'os1-128', 'os1-128_'+lidar_idx+'.pcd')
            path_cam_front = '/'+os.path.join(*path_header, 'cam-front', 'cam-front_'+camf_idx+'.png')
            path_calib = '/'+os.path.join(*path_header, 'info_calib', 'calib_radar_lidar.txt')
            path_desc = '/'+os.path.join(*path_header, 'description.txt')
            path_cube_doppler = None
            if self.is_get_cube_dop:
                if self.is_dop_another_dir:
                    path_cube_doppler = os.path.join(self.dir_dop, path_header[-1], 'radar_cube_doppler', 'radar_cube_doppler_'+radar_idx+'.mat')
                else:
                    path_cube_doppler = '/'+os.path.join(*path_header, 'radar_cube_doppler', 'radar_cube_doppler_'+radar_idx+'.mat')

            meta = {
                'path_label': path_label,
                'seq_id': seq_id,
                'rdr_idx': radar_idx,
                'ldr_idx': lidar_idx,
                'camf_idx': camf_idx,
                'path_rdr_tesseract': path_radar_tesseract,
                'path_rdr_cube': path_radar_cube,
                'path_rdr_bev_img': path_radar_bev_img,
                'path_ldr_bev_img': path_lidar_bev_img,
                'path_ldr_pc_64': path_lidar_pc_64,
                'path_ldr_pc_128': path_lidar_pc_128,
                'path_cam_front': path_cam_front,
                'path_calib': path_calib,
                'path_cube_doppler': path_cube_doppler,
                'path_desc': path_desc
            }

            dic = {
                'meta': meta
            }

            if self.type_coord == 1: # rdr
                dic['calib_info'] = self.get_calib_info(path_calib)
            else: # ldr
                dic['calib_info'] = None
            
            ### Label ###
            dic['meta']['label'] = self.get_label_bboxes(path_label, dic['calib_info'])
            ### Label ###

            ### get only required data ###
            if self.cfg.DATASET.GET_ITEM['rdr_tesseract']:
                dic['rdr_tesseract'] = self.get_tesseract(dic['meta']['path_rdr_tesseract']) 
            if self.cfg.DATASET.GET_ITEM['rdr_cube']:
                if self.cfg.DATASET.RDR_CUBE.USE_PREPROCESSED_CUBE:
                    data_dir = '/'.join(path_radar_cube.split('/')[:-2])
                    cube_name = path_radar_cube.split('/')[-1].split('.')[0]
                    path_sparse_cube = os.path.join(data_dir, 'sparse_cube', cube_name+'.npy')
                    sparse_cube = np.load(path_sparse_cube)

                    dic['sparse_cube'] = sparse_cube[:11520] # 0.9 quantile so that we don't need to make new collate_fn
                    
                else:    
                    rdr_cube, none_zero_mask, rdr_cube_cnt = self.get_cube(dic['meta']['path_rdr_cube'], mode=0)
                    dic['rdr_cube'] = rdr_cube
                    dic['rdr_cube_mask'] = none_zero_mask
                    dic['rdr_cube_cnt'] = rdr_cube_cnt
            if self.is_get_cube_dop:
                if self.cfg.DATASET.GET_ITEM['rdr_cube_doppler']:
                    path_cube_doppler = dic['meta']['path_cube_doppler']
                    dic['rdr_cube_doppler'] = self.get_cube_doppler(path_cube_doppler) if os.path.exists(path_cube_doppler) else None
            else:
                pass
            if self.cfg.DATASET.GET_ITEM['ldr_pc_64']:
                dic['ldr_pc_64'] = self.get_pc_lidar(dic['meta']['path_ldr_pc_64'], dic['calib_info'])

            if self.is_use_gen_labels:
                dic['conf_label'] = self.get_gen_conf_label(dic['meta']['path_label'])


            if self.cfg.DATASET.GET_ITEM['ldr_pc_64'] and self.cfg.DATASET.LIDAR.AUGMENT and (self.split == 'train'):
                sedan_db_path = '/home/oem/Donghee/krdr_0605/object_database/sedan.npy'
                sedan_db = np.load(sedan_db_path, allow_pickle = True)
                random_choice = np.random.randint(0, sedan_db.shape[0], self.cfg.DATASET.LIDAR.NUM_OBJ_SAMPLED)
                sampled_sedan_list = sedan_db[random_choice]

                roi = self.cfg.DATASET.LIDAR.ROI
                x_min, x_max = roi['x']
                y_min, y_max = roi['y']
                z_min, z_max = roi['z']

                min_dx, min_dy = 6.0, 4.0
                label_list = dic['meta']['label']
                obj_pos_list = []
                for label in label_list:
                    cls_name, cls_id, bbox, _ = label
                    xc, yc, zc, theta, xl, yl, zl = bbox
                    obj_pos_list.append([xc, yc])

                num_aug_obj = 0

                for (obj_points, obj_label) in sampled_sedan_list:
                    # New Obj pos
                    cls_name, xl, yl, zl, heading = obj_label

                    # Generate random position
                    new_xc = np.random.uniform()*(x_max - x_min) + x_min
                    new_yc = np.random.uniform()*(y_max - y_min) + y_min

                    # Prevent Collision
                    ok_flag = True
                    for exist_pos in obj_pos_list:
                        exist_xc, exist_yc = exist_pos
                        iter_cnt = 0
                        collision_flag = (abs(new_xc - exist_xc) < min_dx) and (abs(new_yc - exist_yc) < min_dy)
                        if collision_flag:
                            ok_flag = False


                    # Recenter objects
                    if ok_flag:                        
                        obj_points[:, 0] = obj_points[:, 0] + new_xc
                        obj_points[:, 1] = obj_points[:, 1] + new_yc

                        obj_bottom = np.min(obj_points[:, 2])
                        ground_height = np.min(dic['ldr_pc_64'][:, 2])
                        dz = obj_bottom - ground_height
                        obj_points[:, 2] = obj_points[:, 2] - dz
                        obj_zc = np.mean(obj_points[:, 2])

                        new_label = [new_xc, new_yc, obj_zc, heading, xl, yl, zl]

                        dic['ldr_pc_64'] = np.vstack((dic['ldr_pc_64'], obj_points))
                        dic['meta']['label'].append([cls_name, 1, new_label, 1000])
                        obj_pos_list.append([new_xc, new_yc])

                        num_aug_obj += 1               


            # Brute force for sparse cube augmentation
            if self.cfg.DATASET.GET_ITEM['rdr_cube'] and self.cfg.DATASET.RDR_CUBE.AUGMENT and (self.split == 'train'):
                sedan_db_path = '/home/oem/Donghee/krdr_0605/object_database/sedan_radar.npy'
                sedan_db = np.load(sedan_db_path, allow_pickle = True)
                random_choice = np.random.randint(0, sedan_db.shape[0], self.cfg.DATASET.LIDAR.NUM_OBJ_SAMPLED)
                sampled_sedan_list = sedan_db[random_choice]

                roi = self.cfg.DATASET.RDR_CUBE.RDR_CB_ROI
                x_min, x_max = roi['x']
                y_min, y_max = roi['y']
                z_min, z_max = roi['z']

                min_dx, min_dy = 6.0, 4.0
                label_list = dic['meta']['label']
                obj_pos_list = []
                for label in label_list:
                    cls_name, cls_id, bbox, _ = label
                    xc, yc, zc, theta, xl, yl, zl = bbox
                    obj_pos_list.append([xc, yc])

                num_aug_obj = 0

                for (obj_points, obj_label) in sampled_sedan_list:
                    # New Obj pos
                    cls_name, xl, yl, zl, heading = obj_label

                    # Generate random position
                    new_xc = np.random.uniform()*(x_max - x_min) + x_min
                    new_yc = np.random.uniform()*(y_max - y_min) + y_min

                    # Prevent Collision
                    ok_flag = True
                    for exist_pos in obj_pos_list:
                        exist_xc, exist_yc = exist_pos
                        iter_cnt = 0
                        collision_flag = (abs(new_xc - exist_xc) < min_dx) and (abs(new_yc - exist_yc) < min_dy)
                        if collision_flag:
                            ok_flag = False


                    # Recenter objects
                    if ok_flag:                        
                        obj_points[:, 0] = obj_points[:, 0] + new_xc
                        obj_points[:, 1] = obj_points[:, 1] + new_yc

                        obj_bottom = np.min(obj_points[:, 2])
                        ground_height = np.min(dic['sparse_cube'][:, 2])
                        dz = obj_bottom - ground_height
                        obj_points[:, 2] = obj_points[:, 2] - dz
                        obj_zc = np.mean(obj_points[:, 2])

                        new_label = [new_xc, new_yc, obj_zc, heading, xl, yl, zl]

                        dic['sparse_cube'] = np.vstack((dic['sparse_cube'], obj_points))
                        dic['meta']['label'].append([cls_name, 1, new_label, 1000])
                        obj_pos_list.append([new_xc, new_yc])

                        num_aug_obj += 1         

            # sepeartor = ',' without space
            dic['desc'] = self.get_description(path_desc)
            # dic['desc'] = {
            #     'capture_time': 'daylight',
            #     'road_type': 'city',
            #     'climate': 'snowy',
            # }

            # t2 = time.time()
            # print(f"* d0: {t2 - t1:.5f} sec")
            return dic
        except Exception as e:
            ### This is for debugging ###
            pass
            # path_label = self.label_paths[idx]
            # seq_id, radar_idx, lidar_idx, camf_idx = self.get_data_indices(path_label)
            # path_header = path_label.split('/')[:-2]
            # path_radar_cube = '/'+os.path.join(*path_header, 'radar_zyx_cube', 'cube_'+radar_idx+'.mat')
            # print('* error happens !')
            # print(e)
            # print(f'* {path_label}')
            # print(f'* {path_radar_cube}')

            return None
    
    def collate_fn(self, list_dict_batch):
        '''
        * list_dict_batch = list of item (__getitem__)
        '''
        if None in list_dict_batch:
            return None
        
        # t1 = time.time()
        dict_datum = list_dict_batch[0]
        dict_batch = {k: [] for k in dict_datum.keys()} # init empty list per key
        
        dict_batch['labels'] = []
        dict_batch['num_objects'] = []

        for batch_id, dict_temp in enumerate(list_dict_batch):
            for k, v in dict_temp.items():
                if k in ['meta', 'rdr_cube_mask', 'desc']: 
                    dict_batch[k].append(v)
                else:
                    try:
                        dict_batch[k].append(torch.from_numpy(v).float())
                    except:
                        pass

            list_objects = dict_temp['meta']['label']
            num_objects = len(list_objects)
            dict_batch['labels'].append(list_objects)
            dict_batch['num_objects'].append(num_objects)

        for k in dict_datum.keys():
            if not (k == 'meta'):
                if (k == 'ldr_pc_64') or (k == 'ldr_pc_128'):
                    batch_indices = []
                    for batch_id, pc in enumerate(dict_batch[k]):
                        batch_indices.append(torch.full((len(pc),), batch_id))
                    dict_batch[k] = torch.cat(dict_batch[k], dim = 0)
                    dict_batch['pts_batch_indices_'+k] = torch.cat(batch_indices)
                elif k in ['labels', 'num_objects', 'rdr_cube_mask', 'desc']:
                    pass
                else:
                    try:
                        dict_batch[k] = torch.stack(dict_batch[k])
                    except:
                        pass    
        dict_batch['batch_size'] = batch_id+1

        # t2 = time.time()
        # print(f"* d1: {t2 - t1:.5f} sec")

        return dict_batch

if __name__ == '__main__':
    ### temp library ###
    import yaml
    from easydict import EasyDict

    path_cfg = './configs/cfg_total_v2/cfg_total_v2_1/ResNext4D_Cart.yml'
    ### Rdr Cube ###
    
    f = open(path_cfg, 'r')
    try:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    except:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    f.close()

    cfg.DATASET.RDR_CUBE.RDR_CB_ROI = {
        'z': [-2, 5.6], # None (erase)
        'y': [-32, 31.6], # [-9.6, 9.2], # [-6.4, 6.0], # [-32, 31.6],
        'x': [0, 71.6],
    }
    dict_roi = cfg.DATASET.RDR_CUBE.RDR_CB_ROI

    dataset = KRadarDataset_v2_1(cfg=cfg, split='train') # 'train'
    
    idx_datum = 800 # 75: here2
    label = dataset[idx_datum]['meta']['label']
    # print(dataset[idx_datum]['meta'])
    # print(label)
    # print(dataset[idx_datum]['meta'])

    # print(dataset.arr_doppler)

    ### Rdr Tesseract ###
    # dataset.show_radar_tensor_bev(dataset[idx_datum], bboxes=label, roi_x=[0,0.2,100], roi_y=[-80,0.2,80])
    # dataset.show_lidar_point_cloud(dataset[idx_datum], bboxes=label)
    # dataset.show_gaussian_confidence_cart(bboxes=label)
    # dataset.show_gaussian_confidence_polar(bboxes=label)
    # dataset.show_gaussian_confidence_polar_color(bboxes=label)
    # dataset.show_heatmap_polar_with_bbox(idx_datum)
    dataset.show_rdr_pc_tesseract(dataset[idx_datum], bboxes=label, cfar_params=[25,8,0.001],
                                    roi_x=dict_roi['x'], roi_y=dict_roi['y'], roi_z=dict_roi['z'], is_with_lidar=False)
    ### Rdr Tesseract ###

    ### Rdr Cube ###
    # cv2.imshow('front image', cv2.imread(dataset[idx_datum]['meta']['path_cam_front']))
    # cv2.waitKey(0)
    dataset.show_radar_cube_bev(dataset[idx_datum], bboxes=label, is_with_doppler=False, is_with_log=True)
    # dataset.show_sliced_radar_cube(dataset[idx_datum], bboxes=label) # idx_datum = 50: Car shape visualization
    # dataset.show_sliced_radar_cube(dataset[idx_datum], idx_custom_slice=[50, 70, 50, 100]) # idx_datum = 50
    # dataset.show_sliced_radar_cube(dataset[idx_datum], idx_custom_slice=[110, 130, 50, 100]) # idx_datum = 100 (seq 9)
    # dataset.show_sliced_radar_cube(dataset[idx_datum], idx_custom_slice=[30, 50, 50, 100]) # idx_datum = 100 (seq 9)
    # dataset.show_rdr_pc_cube(dataset[idx_datum], bboxes=label, cfar_params=[25,8,0.1], axis='x')
    ### Rdr Cube ###
    