'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

from cv2 import batchDistance
import torch
import torch.nn as nn
from spconv.pytorch.utils import PointToVoxel
import numpy as np

class MeanVoxelEncoder_Radar_withDop(nn.Module):
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

        self.rdr_voxel_size = self.cfg.DATASET.RDR_CUBE.GRID_SIZE
        self.rdr_pc_range = self.cfg.DATASET.RDR_CUBE.RDR_CB_ROI
        self.rdr_num_point_features = self.cfg.DATASET.RDR_CUBE.NUM_POINT_FEATURES
        self.rdr_max_num_voxels = 16000
        self.rdr_max_num_pts_per_voxels = 3

        rdr_pc_range = [
            self.rdr_pc_range['x'][0], self.rdr_pc_range['y'][0], self.rdr_pc_range['z'][0],
            self.rdr_pc_range['x'][1], self.rdr_pc_range['y'][1], self.rdr_pc_range['z'][1],
        ]

        self.gen_voxels = PointToVoxel(
            vsize_xyz = [self.rdr_voxel_size for _ in range(3)],
            coors_range_xyz = rdr_pc_range,
            num_point_features = 5,
            max_num_voxels = self.rdr_max_num_voxels,
            max_num_points_per_voxel = self.rdr_max_num_pts_per_voxels,
            device = torch.device('cuda')  # Assuming no distributed training
        )

    def forward(self, data_dic, **kwargs):
        """
        Args:
            data_dic:
                voxels: num_voxels x max_points_per_voxel x C_points
                voxel_num_points: optional (num_voxels)

        Returns:
            vfe_features: (num_voxels, C)
        """

        batch_voxel_features, batch_voxel_coords, batch_num_pts_in_voxels = [], [], []

        if self.cfg.DATASET.RDR_CUBE.USE_PREPROCESSED_CUBE:
            if self.cfg.DATASET.RDR_CUBE.AUGMENT:
                rdr_cube = []
                for batch_id, rdr_cube_sample in enumerate(data_dic['sparse_cube']):
                    rdr_cube_sample = rdr_cube_sample.cuda()
                   
                    batch_ids = torch.full((rdr_cube_sample.shape[0], 1), batch_id, dtype = rdr_cube_sample.dtype, device = rdr_cube_sample.device)

                    voxel_features, voxel_coords, voxel_num_points = self.gen_voxels(rdr_cube_sample)
                    batch_ids_voxel = torch.full((voxel_coords.shape[0], 1), batch_id, dtype = voxel_coords.dtype, device = voxel_coords.device)
                    voxel_coords = torch.cat((batch_ids_voxel, voxel_coords), dim = -1)

                    batch_voxel_features.append(voxel_features)
                    batch_voxel_coords.append(voxel_coords)
                    batch_num_pts_in_voxels.append(voxel_num_points)

                    rdr_cube_sample_with_batch = torch.cat((batch_ids, rdr_cube_sample), dim = -1)
                    rdr_cube.append(rdr_cube_sample_with_batch)
                

                rdr_cube = torch.cat(rdr_cube)
                data_dic['points'] = rdr_cube.cuda()

            else:
                rdr_cube = data_dic['sparse_cube'].cuda() # B N 4 --> No Augment
                rdr_cube_dop = data_dic['sparse_cube_dop'].cuda()

                cat_rdr_cube = torch.cat((rdr_cube, rdr_cube_dop[:,:,3:4]), dim = -1)

                for batch_id in range(data_dic['batch_size']):
                    pc = cat_rdr_cube[batch_id]
                    voxel_features, voxel_coords, voxel_num_points = self.gen_voxels(pc)
                    voxel_batch_id = torch.full((voxel_coords.shape[0], 1), batch_id, device = rdr_cube.device, dtype = torch.int64)
                    voxel_coords = torch.cat((voxel_batch_id, voxel_coords), dim = -1)
                    
                    batch_voxel_features.append(voxel_features)
                    batch_voxel_coords.append(voxel_coords)
                    batch_num_pts_in_voxels.append(voxel_num_points)

                ## Additional Points Preprocessing for PVRCN_PP ##
                B, N, C = cat_rdr_cube.shape
                pts = cat_rdr_cube.view(B*N, C)
                pts_indices = torch.arange(0, B, step=1).view(B, 1).repeat(1, N).view(B*N).to(pts.device)
                pts_coords = torch.cat((pts_indices.unsqueeze(1), pts), dim = -1)
                # data_dic['point_coords'] = pts_coords[:, :4].cuda() # N x (batch_idx, x, y, z)
                data_dic['points'] = pts_coords.cuda() # N x (batch_ind, x, y, z, C)
                

        else:
            rdr_cube = data_dic['rdr_cube'].cuda()

            batch_indices = []
            radar_pc = []
            z_min, z_max = self.rdr_pc_range['z']
            y_min, y_max = self.rdr_pc_range['y']
            x_min, x_max = self.rdr_pc_range['x']

            for sample_idx in range(data_dic['batch_size']):
                sample_rdr_cube = rdr_cube[sample_idx]
                z_ind, y_ind, x_ind = torch.where(sample_rdr_cube > sample_rdr_cube.quantile(0.9))
                
                power_val = sample_rdr_cube[z_ind, y_ind, x_ind].unsqueeze(-1)
                power_val = power_val / 1e+13 # Heuristic normalization

                z_coord = (z_ind/sample_rdr_cube.shape[0]*(z_max-z_min)+z_min).unsqueeze(-1)
                y_coord = (y_ind/sample_rdr_cube.shape[1]*(y_max-y_min)+y_min).unsqueeze(-1)
                x_coord = (x_ind/sample_rdr_cube.shape[2]*(x_max-x_min)+x_min).unsqueeze(-1)
                
                sparse_rdr_cube = torch.cat((x_coord, y_coord, z_coord, power_val), dim=-1) # N, 4
                radar_pc.append(sparse_rdr_cube)

                sample_indices = torch.full((sparse_rdr_cube.shape[0],1), sample_idx, device = sparse_rdr_cube.device)
                sample_indices = torch.cat((sample_indices, z_ind.unsqueeze(-1), y_ind.unsqueeze(-1), x_ind.unsqueeze(-1)), dim = -1)

                voxel_features, voxel_coords, voxel_num_points = self.gen_voxels(sparse_rdr_cube)
                voxel_batch_id = torch.full((voxel_coords.shape[0], 1), sample_idx, device = rdr_cube.device, dtype = torch.int64)
                voxel_coords = torch.cat((voxel_batch_id, voxel_coords), dim = -1)
                
                batch_voxel_features.append(voxel_features)
                batch_voxel_coords.append(voxel_coords)
                batch_num_pts_in_voxels.append(voxel_num_points)
        
                batch_indices.append(torch.full((len(power_val), 1), sample_idx, device = voxel_features.device))            


            pts_coords = torch.cat((torch.cat(batch_indices), torch.cat(radar_pc)), dim = -1)
            data_dic['points'] = pts_coords.cuda() # N x (batch_ind, x, y, z, C)
            

        voxel_features, voxel_coords, voxel_num_points = torch.cat(batch_voxel_features), torch.cat(batch_voxel_coords), torch.cat(batch_num_pts_in_voxels)
        data_dic['voxel_features'], data_dic['voxel_coords'], data_dic['voxel_num_points'] = voxel_features, voxel_coords, voxel_num_points
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        encoded_voxels = points_mean / normalizer # BM x C

        # for encoder in self.voxel_encoder:
        #     encoded_voxels = encoder(encoded_voxels)
            
        data_dic['encoded_voxel_features'] = encoded_voxels.contiguous()
    
        return data_dic
