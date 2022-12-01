'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import torch
import torch.nn as nn

class RadarCubeSparseProcessor(nn.Module):
    def __init__(self, cfg):
        super(RadarCubeSparseProcessor, self).__init__()
        self.cfg = cfg
        self.roi = cfg.DATASET.RDR_CUBE.RDR_CB_ROI
        self.mode = cfg.MIDDLE_ENCODER.MODE

    def forward(self, dict_datum):
        rdr_cube = dict_datum['rdr_cube'].cuda()
        
        z_min, z_max = self.roi['z']
        y_min, y_max = self.roi['y']
        x_min, x_max = self.roi['x']

        sparse_radar = []
        sample_ind = []

        for sample_idx in range(dict_datum['batch_size']):
            sample_rdr_cube = rdr_cube[sample_idx]
            z_ind, y_ind, x_ind = torch.where(sample_rdr_cube > sample_rdr_cube.quantile(0.7))
            
            power_val = sample_rdr_cube[z_ind, y_ind, x_ind].unsqueeze(-1)
            power_val = power_val / 1e+13 # Heuristic normalization

            z_coord = (z_ind/sample_rdr_cube.shape[0]*(z_max-z_min)+z_min).unsqueeze(-1)
            y_coord = (y_ind/sample_rdr_cube.shape[1]*(y_max-y_min)+y_min).unsqueeze(-1)
            x_coord = (x_ind/sample_rdr_cube.shape[2]*(x_max-x_min)+x_min).unsqueeze(-1)
            
            sparse_rdr_cube = torch.cat((x_coord, y_coord, z_coord, power_val), dim=-1) # N, 4
            sample_indices = torch.full((sparse_rdr_cube.shape[0],1), sample_idx, device = sparse_rdr_cube.device)
            sample_indices = torch.cat((sample_indices, z_ind.unsqueeze(-1), y_ind.unsqueeze(-1), x_ind.unsqueeze(-1)), dim = -1)

            sparse_radar.append(sparse_rdr_cube)
            sample_ind.append(sample_indices)
        
        batch_sparse_radar = torch.cat(sparse_radar, dim = 0)
        batch_ind = torch.cat(sample_ind, dim = 0)

        dict_datum['sparse_features'] = batch_sparse_radar
        dict_datum['sparse_indices'] = batch_ind


        return dict_datum
