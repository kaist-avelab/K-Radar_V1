"""
# -*- coding: utf-8 -*-
--------------------------------------------------------------------------------
# author: Donghee Paek, AVELab, KAIST
# date:   2022.05.29
# e-mail: donghee.paek@kaist.ac.kr
--------------------------------------------------------------------------------
# description: script for vis
"""

# Library
import sys
import os
from PyQt5 import QtGui
import numpy as np
import cv2
import open3d as o3d
import yaml
from easydict import EasyDict
from tqdm import tqdm

from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

# User Library
import datasets
from utils.util_ui_vis import *
from utils.util_geometry import Object3D

path_ui = '%s/uis/ui_vis.ui' % '.' # cnf.BASE_DIR
class_ui = uic.loadUiType(path_ui)[0]

class MainFrame(QMainWindow, class_ui):
    def __init__(self, cfg):
        super().__init__()
        self.setupUi(self)
        self.cfg = cfg
        self.init_signals()
        self.kradar = None
        self.dict_datum = None
        
    def init_signals(self):
        # Initialize signals...
        list_name_fuction = [
            'pushButtonLoad',               # 0
            'pushButtonCalibrate',          # 1
            'pushButtonCameraVis',
            'pushButtonLidarVis',
            'pushButtonRadarVis',
            ]

        for i in range(len(list_name_fuction)):
            getattr(self, f'pushButton_{i}').clicked.\
                connect(getattr(self, list_name_fuction[i]))

        self.listWidget_files.itemDoubleClicked.connect(self.listWidget_files_doubleClicked)

    def pushButtonLoad(self):
        ### Change to cfg ###
        self.split = 'train' if self.radioButton_training.isChecked() else 'test'
        self.cfg.DATASET.RDR_CUBE.RDR_CB_ROI['x'] = [0, 120]
        self.cfg.DATASET.RDR_CUBE.RDR_CB_ROI['y'] = [-100, 100]
        self.cfg.DATASET.RDR_CUBE.RDR_CB_ROI['z'] = [-50, 50]

        self.kradar = datasets.__all__[self.cfg.DATASET.NAME](cfg=self.cfg, split=self.split)

        ### For visualization ###
        self.kradar.is_roi_check_with_azimuth = False

        # Add to list widget
        self.listWidget_files.clear()

        dict_label_dist = {
                'Sedan':0,
                'Bus or Truck':0,
                'Motorcycle':0,
                'Bicycle':0,
                'Pedestrian':0,
                'Pedestrian Group':0,
            }
        dict_meter_dist = {
                '10':0,
                '20':0,
                '30':0,
                '40':0,
                '50':0,
                '60':0,
                '70':0,
                '80':0,
                '90':0,
                '100':0,
                '110':0,
                '120':0,
                '130':0,
                '140':0,
                '150':0,
            }
        list_meter = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150']
        
        for idx_label, path_label in tqdm(enumerate(self.kradar.label_paths)):
            seq_id, radar_idx, lidar_idx, camf_idx = self.kradar.get_data_indices(path_label)
            path_header = path_label.split('/')[:-2]
            seq = path_header[-1]
            path_radar_tesseract = '/'+os.path.join(*path_header, 'radar_tesseract', 'tesseract_'+radar_idx+'.mat')
            path_radar_cube = '/'+os.path.join(*path_header, 'radar_zyx_cube', 'cube_'+radar_idx+'.mat')
            path_radar_bev_img = '/'+os.path.join(*path_header, 'radar_bev_image', 'radar_bev_100_'+radar_idx+'.png')
            path_lidar_bev_img = '/'+os.path.join(*path_header, 'lidar_bev_image', 'lidar_bev_100_'+lidar_idx+'.png')
            path_lidar_pc_64 = '/'+os.path.join(*path_header, 'os2-64', 'os2-64_'+lidar_idx+'.pcd')
            path_lidar_pc_128 = '/'+os.path.join(*path_header, 'os1-128', 'os1-128_'+lidar_idx+'.pcd')
            path_cam_front = '/'+os.path.join(*path_header, 'cam-front', 'cam-front_'+camf_idx+'.png')
            path_calib = '/'+os.path.join(*path_header, 'info_calib', 'calib_radar_lidar.txt')
            path_desc = '/'+os.path.join(*path_header, 'description.txt')

            dict_datum = dict()

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
                'path_desc': path_desc,
            }

            dict_datum['meta'] = meta

            ### Process desc (TBD) ###
            dict_datum['desc'] = self.kradar.get_description(path_desc)
            cap_time = dict_datum['desc']['capture_time']
            road_type = dict_datum['desc']['road_type']
            climate = dict_datum['desc']['climate']
            ### Process desc (TBD) ###

            if self.kradar.type_coord == 1: # rdr
                dict_datum['calib_info'] = self.kradar.get_calib_info(path_calib)
            else: # ldr
                dict_datum['calib_info'] = None
            
            ### Label ###
            dict_datum['meta']['label'] = self.kradar.get_label_bboxes(path_label, dict_datum['calib_info'])
            ### Label ###

            ### Calculate label distribution ###
            for tuple_obj in dict_datum['meta']['label']:
                cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj = tuple_obj
                dict_label_dist[cls_name] += 1
                dist = np.sqrt(x**2+y**2+z**2)
                # print(dist)
                dist_key_name = list_meter[int(dist/10)]
                # print(dist_key_name)
                dict_meter_dist[dist_key_name] += 1
            ### Calculate label distribution ###

            with open(path_label, 'r') as f:
                lines = f.readlines()
                f.close()
            tstamp = np.round(float(lines[0].split(',')[1].split('=')[1]), decimals=2)

            temp_item = QListWidgetItem()
            temp_item.setData(1, dict_datum)
            temp_item.setText(str(idx_label) + '. ' + f'seq_{seq}: {tstamp} / {cap_time} / {road_type} / {climate}')
            self.listWidget_files.addItem(temp_item)

        for k, v in dict_label_dist.items():
            print(f'{k}: {v}')

        for k, v in dict_meter_dist.items():
            print(f'{k}: {v}')


    def listWidget_files_doubleClicked(self):
        current_item = self.listWidget_files.currentItem()
        self.textBrowser_logs.append(current_item.data(0) + ' is loaded')
        self.dict_datum = current_item.data(1)

        cv_img = cv2.imread(self.dict_datum['meta']['path_cam_front'])
        cv_img = cv_img[:,1280:,:] if self.checkBox_frontCam.isChecked() else cv_img[:,:1280,:]
        temp_front_img = get_q_pixmap_from_cv_img(cv_img, 768, 432)
        getattr(self, 'label_frontImg').setPixmap(temp_front_img)
        
        labels = self.dict_datum['meta']['label']
        print(labels)

    def pushButtonCalibrate(self):
        if self.dict_datum is None:
            print('* select at least one item')
            return
        
        cv_img = cv2.imread(self.dict_datum['meta']['path_cam_front'])
        cv_img = cv_img[:,1280:,:] if self.checkBox_frontCam.isChecked() else cv_img[:,:1280,:]
        cv_img_ori = cv_img.copy()

        intrinsic, extrinsic = get_intrinsic_and_extrinsic_params_from_text_edit(get_list_p_text_edit(self))
        rot, tra = get_rotation_and_translation_from_extrinsic(extrinsic)

        labels = self.dict_datum['meta']['label']
        list_objs = []
        
        list_line_order = [[0,1], [0,2], [1,3], [2, 3], [0,4], [1,5], [2,6], [3,7], [4,5], [4,6], [5,7], [6,7]]
        for tuple_obj in labels:
            cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj = tuple_obj
            arr_points = Object3D(x, y, z, l, w, h, theta).corners
            arr_points = get_pointcloud_with_rotation_and_translation(arr_points, rot, tra)
            arr_pix = get_pixel_from_point_cloud_in_camera_coordinate(arr_points, intrinsic)
            list_objs.append(arr_pix)
            for idx_1, idx_2 in list_line_order:
                p1_x, p1_y = arr_pix[idx_1]
                p2_x, p2_y = arr_pix[idx_2]
                p1_x = int(np.round(p1_x))
                p1_y = int(np.round(p1_y))
                p2_x = int(np.round(p2_x))
                p2_y = int(np.round(p2_y))

                if self.checkBox_color.isChecked():
                    color = self.cfg.VIS.DIC_CLASS_BGR[cls_name]
                else:
                    color = (0, 255, 0)
                
                cv_img = cv2.line(cv_img, (p1_x,p1_y), (p2_x,p2_y), color, thickness=2)
        
        alpha = 0.5
        cv_img = cv2.addWeighted(cv_img, alpha, cv_img_ori, 1 - alpha, 0)
        temp_front_img = get_q_pixmap_from_cv_img(cv_img, 768, 432)
        getattr(self, 'label_frontImg').setPixmap(temp_front_img)

    def pushButtonCameraVis(self):
        if self.dict_datum is None:
            print('* select at least one item')
            return
        
        cv_img = cv2.imread(self.dict_datum['meta']['path_cam_front'])
        cv_img = cv_img[:,1280:,:] if self.checkBox_frontCam.isChecked() else cv_img[:,:1280,:]
        cv_img_ori = cv_img.copy()

        if self.checkBox_bbox.isChecked():
            intrinsic, extrinsic = get_intrinsic_and_extrinsic_params_from_text_edit(get_list_p_text_edit(self))
            rot, tra = get_rotation_and_translation_from_extrinsic(extrinsic)

            labels = self.dict_datum['meta']['label']
            list_objs = []
            
            list_line_order = [[0,1], [0,2], [1,3], [2, 3], [0,4], [1,5], [2,6], [3,7], [4,5], [4,6], [5,7], [6,7]]
            for tuple_obj in labels:
                cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj = tuple_obj
                arr_points = Object3D(x, y, z, l, w, h, theta).corners
                arr_points = get_pointcloud_with_rotation_and_translation(arr_points, rot, tra)
                arr_pix = get_pixel_from_point_cloud_in_camera_coordinate(arr_points, intrinsic)
                list_objs.append(arr_pix)
                for idx_1, idx_2 in list_line_order:
                    p1_x, p1_y = arr_pix[idx_1]
                    p2_x, p2_y = arr_pix[idx_2]
                    p1_x = int(np.round(p1_x))
                    p1_y = int(np.round(p1_y))
                    p2_x = int(np.round(p2_x))
                    p2_y = int(np.round(p2_y))

                    if self.checkBox_color.isChecked():
                        color = self.cfg.VIS.DIC_CLASS_BGR[cls_name]
                    else:
                        color = (0, 255, 0)
                    
                    cv_img = cv2.line(cv_img, (p1_x,p1_y), (p2_x,p2_y), color, thickness=2)

        # temp_front_img = get_q_pixmap_from_cv_img(cv_img, 768, 432)
        # getattr(self, 'label_frontImg').setPixmap(temp_front_img)

        alpha = 0.5
        cv_img = cv2.addWeighted(cv_img, alpha, cv_img_ori, 1 - alpha, 0)
        cv2.imshow('front_iamge', cv_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def pushButtonLidarVis(self):
        if self.dict_datum is None:
            print('* select at least one item')
            return

        self.dict_datum['ldr_pc_64'] = \
            self.kradar.get_pc_lidar(self.dict_datum['meta']['path_ldr_pc_64'], self.dict_datum['calib_info'])
        print(self.dict_datum['calib_info'])

        if self.checkBox_bbox.isChecked():
            bboxes = self.dict_datum['meta']['label']
        else:
            bboxes = None

        if self.checkBox_rdr_pc.isChecked():
            is_with_rdr_pc = True
            cfar_params = self.plainTextEdit_cfar.toPlainText()
            cfar_params = cfar_params.split(',')
            cfar_params = [int(cfar_params[0]), int(cfar_params[1]), float(cfar_params[2])]
            self.dict_datum['rdr_tesseract'] = self.kradar.get_tesseract(self.dict_datum['meta']['path_rdr_tesseract']) 
        else:
            is_with_rdr_pc = False
            cfar_params = [25,8,0.01]
        
        self.kradar.show_lidar_point_cloud(
            self.dict_datum, bboxes, \
            roi_x=[0, 150], roi_y=[-60, 60], roi_z=[-10.0, 10],
            roi_x_rdr=[0, 100], roi_y_rdr=[-50, 50], roi_z_rdr=[-2.0, 5],
        )

    def pushButtonRadarVis(self):
        if self.dict_datum is None:
            print('* select at least one item')
            return

        # self.dict_datum = 
        self.dict_datum['rdr_tesseract'] = self.kradar.get_tesseract(self.dict_datum['meta']['path_rdr_tesseract'])

        if self.checkBox_bbox.isChecked():
            bboxes = self.dict_datum['meta']['label']
        else:
            bboxes = None

        self.kradar.show_radar_tensor_bev(self.dict_datum, bboxes, roi_x = [0, 0.4, 80], roi_y = [-60, 0.4, 60])
        
def startUi(path_cfg):
    f = open(path_cfg, 'r')
    try:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    except:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    f.close()

    app = QApplication(sys.argv)
    main_frame = MainFrame(cfg)
    main_frame.show()
    sys.exit(app.exec_())
