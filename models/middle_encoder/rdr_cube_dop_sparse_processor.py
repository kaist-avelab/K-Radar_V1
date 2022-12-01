'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

from turtle import ycor
from numpy import single
import torch
import torch.nn as nn
from spconv.pytorch.utils import PointToVoxel

class RadarDopSparseProcessor(nn.Module):
    def __init__(self, cfg):
        super(RadarDopSparseProcessor, self).__init__()
        self.cfg = cfg
        self.roi = cfg.DATASET.RDR_CUBE.RDR_CB_ROI

        z_min, z_max = self.roi['z']
        y_min, y_max = self.roi['y']
        x_min, x_max = self.roi['x']

        self.gen = PointToVoxel(
            vsize_xyz = [0.4, 0.4, 0.4],
            coors_range_xyz = [x_min, y_min, z_min, x_max, y_max, z_max],
            num_point_features=1,
            max_num_voxels=40000,
            max_num_points_per_voxel=5
        )

    def forward(self, dict_datum):
        if self.cfg.DATASET.RDR_CUBE.USE_PREPROCESSED_CUBE:
            sparse_rdr_cube_pow = dict_datum['sparse_cube'].cuda()
            sparse_rdr_cube_dop = dict_datum['sparse_cube_dop'].cuda()
            sparse_rdr_cube = torch.cat((sparse_rdr_cube_pow[:, :, 0:4], sparse_rdr_cube_dop[:, :, 3:4]), dim = -1)

            B, N, C = sparse_rdr_cube.shape
            batch_indices_list = []
            for batch_idx in range(B):
                batch_indices = torch.full((N,1), batch_idx, dtype = torch.long).cuda()
                batch_indices_list.append(batch_indices)
            batch_indices_list = torch.cat(batch_indices_list)

            sparse_rdr_cube = sparse_rdr_cube.view(B*N, C)

            z_min, z_max = self.roi['z']
            y_min, y_max = self.roi['y']
            x_min, x_max = self.roi['x']

            grid_size = self.cfg.DATASET.RDR_CUBE.GRID_SIZE

            x_coord, y_coord, z_coord = sparse_rdr_cube[:, 0:1], sparse_rdr_cube[:, 1:2], sparse_rdr_cube[:, 2:3]
            z_ind = torch.floor((z_coord-z_min) / grid_size).long()
            y_ind = torch.floor((y_coord-y_min) / grid_size).long()
            x_ind = torch.floor((x_coord-x_min) / grid_size).long()

            batch_indices_list = torch.cat((batch_indices_list, z_ind, y_ind, x_ind), dim = -1)
            dict_datum['sparse_features'] = sparse_rdr_cube
            dict_datum['sparse_indices'] = batch_indices_list
        else:
            rdr_cube = dict_datum['rdr_cube'].cuda()
            rdr_cube_dop = dict_datum['rdr_cube_doppler'].cuda()

            z_min, z_max = self.roi['z']
            y_min, y_max = self.roi['y']
            x_min, x_max = self.roi['x']

            sparse_radar = []
            sample_ind = []

            for sample_idx in range(dict_datum['batch_size']):
                sample_rdr_cube = rdr_cube[sample_idx]
                sample_rdr_cube_dop = rdr_cube_dop[sample_idx]
                z_ind, y_ind, x_ind = torch.where(sample_rdr_cube > sample_rdr_cube.quantile(0.9))
                
                power_val = sample_rdr_cube[z_ind, y_ind, x_ind].unsqueeze(-1)
                power_val = power_val / 1e+13 # Heuristic normalization
                dop_val = sample_rdr_cube_dop[z_ind, y_ind, x_ind].unsqueeze(-1) - 1.9326 # from normalize

                z_coord = (z_ind/sample_rdr_cube.shape[0]*(z_max-z_min)).unsqueeze(-1)
                y_coord = (y_ind/sample_rdr_cube.shape[1]*(y_max-y_min)).unsqueeze(-1)
                x_coord = (x_ind/sample_rdr_cube.shape[2]*(x_max-x_min)).unsqueeze(-1)
                
                sparse_rdr_cube = torch.cat((x_coord, y_coord, z_coord, power_val, dop_val), dim=-1) # N, 4
                sample_indices = torch.full((sparse_rdr_cube.shape[0],1), sample_idx, device = sparse_rdr_cube.device)
                sample_indices = torch.cat((sample_indices, z_ind.unsqueeze(-1), y_ind.unsqueeze(-1), x_ind.unsqueeze(-1)), dim = -1)

                sparse_radar.append(sparse_rdr_cube)
                sample_ind.append(sample_indices)
            
            batch_sparse_radar = torch.cat(sparse_radar, dim = 0)
            batch_ind = torch.cat(sample_ind, dim = 0)

            dict_datum['sparse_features'] = batch_sparse_radar
            dict_datum['sparse_indices'] = batch_ind


        return dict_datum
