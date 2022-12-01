'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import torch
import torch.nn as nn
from spconv.pytorch.utils import PointToVoxel
import numpy as np

class MeanVoxelEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_point_features = cfg.MODEL.VOXEL_ENCODER.NUM_POINT_FEATURES
        self.num_filters = cfg.MODEL.VOXEL_ENCODER.NUM_FILTERS

        io_channels = [self.num_point_features] + self.num_filters

        self.voxel_encoder = nn.ModuleList()
        for i in range(len(io_channels)-1):
            self.voxel_encoder.append(nn.Sequential(
                nn.Linear(io_channels[i], io_channels[i+1]),
                nn.ReLU()
            ))

        self.ldr_voxel_size = self.cfg.MODEL.VOXEL_ENCODER.LDR_VOXEL_SIZE
        self.ldr_pc_range = self.cfg.MODEL.VOXEL_ENCODER.LDR_PC_RANGE
        self.ldr_num_point_features = self.cfg.MODEL.VOXEL_ENCODER.LDR_NUM_POINT_FEATURES
        self.ldr_max_num_voxels = self.cfg.MODEL.VOXEL_ENCODER.LDR_MAX_NUM_VOXELS
        self.ldr_max_num_pts_per_voxels = self.cfg.MODEL.VOXEL_ENCODER.LDR_MAX_NUM_PTS_PER_VOXELS

        # self.gen_voxels = PointToVoxel(
        #     vsize_xyz = cfg.DATASET.LDR_VOXEL_SIZE,
        #     coors_range_xyz = cfg.DATASET.LDR_PC_RANGE,
        #     num_point_features = cfg.DATASET.LDR_NUM_POINT_FEATURES,
        #     max_num_voxels = cfg.DATASET.LDR_MAX_NUM_VOXELS,
        #     max_num_points_per_voxel = cfg.DATASET.LDR_MAX_NUM_PTS_PER_VOXELS,
        #     device = torch.device('cuda') # Assuming no distributed training
        # )

    def forward(self, data_dic, **kwargs):
        """
        Args:
            data_dic:
                voxels: num_voxels x max_points_per_voxel x C_points
                voxel_num_points: optional (num_voxels)

        Returns:
            vfe_features: (num_voxels, C)
        """
        ldr_pc_64 = data_dic['ldr_pc_64'].cuda()
        pts_batch_indices = data_dic['pts_batch_indices_ldr_pc_64'].cuda()

        # due to multi-gpu problem
        self.gen_voxels = PointToVoxel(
            vsize_xyz = self.ldr_voxel_size,
            coors_range_xyz = self.ldr_pc_range,
            num_point_features = self.ldr_num_point_features,
            max_num_voxels = self.ldr_max_num_voxels,
            max_num_points_per_voxel = self.ldr_max_num_pts_per_voxels,
            device = ldr_pc_64.device # Assuming no distributed training
        )
        
        
        batch_voxel_features, batch_voxel_coords, batch_num_pts_in_voxels = [], [], []
        for batch_id in range(data_dic['batch_size']):
            pc = ldr_pc_64[torch.where(pts_batch_indices == batch_id)].view(-1, 4)
            voxel_features, voxel_coords, voxel_num_points = self.gen_voxels(pc)
            voxel_batch_id = torch.full((voxel_coords.shape[0], 1), batch_id, device = ldr_pc_64.device, dtype = torch.int64)
            voxel_coords = torch.cat((voxel_batch_id, voxel_coords), dim = -1)
            
            batch_voxel_features.append(voxel_features)
            batch_voxel_coords.append(voxel_coords)
            batch_num_pts_in_voxels.append(voxel_num_points)

        voxel_features, voxel_coords, voxel_num_points = torch.cat(batch_voxel_features), torch.cat(batch_voxel_coords), torch.cat(batch_num_pts_in_voxels)
        data_dic['voxel_features'], data_dic['voxel_coords'], data_dic['voxel_num_points'] = voxel_features, voxel_coords, voxel_num_points
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        encoded_voxels = points_mean / normalizer # BM x C

        # for encoder in self.voxel_encoder:
        #     encoded_voxels = encoder(encoded_voxels)
            
        data_dic['encoded_voxel_features'] = encoded_voxels.contiguous()

        ## Additional Points Preprocessing for PVRCN_PP ##
        pts = data_dic['ldr_pc_64']
        pts_indices = data_dic['pts_batch_indices_ldr_pc_64']
        pts_coords = torch.cat((pts_indices.unsqueeze(1), pts), dim = -1)
        data_dic['point_coords'] = pts_coords[:, :4].cuda()
        data_dic['points'] = pts_coords.cuda() # N x (batch_ind, x, y, z, C)

        return data_dic
