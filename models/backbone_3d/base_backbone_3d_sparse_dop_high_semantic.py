import torch
import torch.nn as nn

import spconv.pytorch as spconv

class BaseBackbone3DSparseModule(nn.Module):
    def __init__(self, cfg):
        super(BaseBackbone3DSparseModule, self).__init__()
        self.cfg = cfg
        self.input_conv = spconv.SparseConv3d(in_channels = 4, out_channels = 64, kernel_size = 1, stride = 1, padding = 0, dilation = 1, indice_key = 'sp0')

        self.spconv1 = spconv.SparseConv3d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 0, dilation = 1, indice_key = 'sp1')
        self.bn1 = nn.BatchNorm1d(64)
        self.subm1a = spconv.SubMConv3d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 0, dilation = 1, indice_key='subm1')
        self.bn1a = nn.BatchNorm1d(64)
        self.subm1b = spconv.SubMConv3d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 0, dilation = 1, indice_key='subm1')
        self.bn1b = nn.BatchNorm1d(64)

        self.spconv2 = spconv.SparseConv3d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 0, dilation = 1, indice_key = 'sp2')
        self.bn2 = nn.BatchNorm1d(128)
        self.subm2a = spconv.SubMConv3d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 0, dilation = 1, indice_key='subm2')
        self.bn2a = nn.BatchNorm1d(128)
        self.subm2b = spconv.SubMConv3d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 0, dilation = 1, indice_key='subm2')
        self.bn2b = nn.BatchNorm1d(128)

        self.spconv3 = spconv.SparseConv3d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 2, padding = 0, dilation = 1, indice_key = 'sp3')
        self.bn3 = nn.BatchNorm1d(256)
        self.subm3a = spconv.SubMConv3d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 0, dilation = 1, indice_key='subm3')
        self.bn3a = nn.BatchNorm1d(256)
        self.subm3b = spconv.SubMConv3d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 0, dilation = 1, indice_key='subm3')
        self.bn3b = nn.BatchNorm1d(256)

        self.toBEV1 = spconv.SparseConv3d(in_channels=64, out_channels=64, kernel_size=(18, 1, 1))
        self.bnBEV1 = nn.BatchNorm1d(64)

        self.toBEV2 = spconv.SparseConv3d(in_channels=128, out_channels=128, kernel_size=(8, 1, 1))
        self.bnBEV2 = nn.BatchNorm1d(128)

        self.toBEV3 = spconv.SparseConv3d(in_channels=256, out_channels=256, kernel_size=(3, 1, 1))
        self.bnBEV3 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, input_sp_tensor):
        x = self.input_conv(input_sp_tensor)

        x = self.spconv1(x)
        x = x.replace_feature(self.bn1(x.features))
        x = x.replace_feature(self.relu(x.features))
        x = self.subm1a(x)
        x = x.replace_feature(self.bn1a(x.features))
        x = x.replace_feature(self.relu(x.features))
        x = self.subm1b(x)
        x = x.replace_feature(self.bn1b(x.features))
        x = x.replace_feature(self.relu(x.features))
        bev_1 = self.toBEV1(x)
        bev_1 = bev_1.replace_feature(self.bnBEV1(bev_1.features))
        bev_1 = bev_1.replace_feature(self.relu(bev_1.features))

        x = self.spconv2(x)
        x = x.replace_feature(self.bn2(x.features))
        x = x.replace_feature(self.relu(x.features))
        x = self.subm2a(x)
        x = x.replace_feature(self.bn2a(x.features))
        x = x.replace_feature(self.relu(x.features))
        x = self.subm2b(x)
        x = x.replace_feature(self.bn2b(x.features))
        x = x.replace_feature(self.relu(x.features))
        bev_2 = self.toBEV2(x)
        bev_2 = bev_2.replace_feature(self.bnBEV2(bev_2.features))
        bev_2 = bev_2.replace_feature(self.relu(bev_2.features))

        x = self.spconv3(x)
        x = x.replace_feature(self.bn3(x.features))
        x = x.replace_feature(self.relu(x.features))
        x = self.subm3a(x)
        x = x.replace_feature(self.bn3a(x.features))
        x = x.replace_feature(self.relu(x.features))
        x = self.subm3b(x)
        x = x.replace_feature(self.bn3b(x.features))
        x = x.replace_feature(self.relu(x.features))
        bev_3 = self.toBEV3(x)
        bev_3 = bev_3.replace_feature(self.bnBEV3(bev_3.features))
        bev_3 = bev_3.replace_feature(self.relu(bev_3.features))
        
        return [bev_1.dense().squeeze(2), bev_2.dense().squeeze(2), bev_3.dense().squeeze(2)]

import math

class BaseBackbone3DSparseDopHSemantic(nn.Module):
    def __init__(self, cfg):
        super(BaseBackbone3DSparseDopHSemantic, self).__init__()
        self.cfg = cfg
        self.sham_pw = BaseBackbone3DSparseModule(cfg)
        self.sham_dop = BaseBackbone3DSparseModule(cfg)

        self.relu = nn.ReLU()

        self.convtrans2d_1 = nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.bnt1 = nn.BatchNorm2d(256)

        self.convtrans2d_2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=6, stride=2)
        self.bnt2 = nn.BatchNorm2d(256)

        self.convtrans2d_3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=12, stride=4)
        self.bnt3 = nn.BatchNorm2d(256)

    def forward(self, data_dic):
        roi = self.cfg.DATASET.RDR_CUBE.RDR_CB_ROI
        grid_size = self.cfg.DATASET.RDR_CUBE.GRID_SIZE
        z_shape = int(math.ceil((roi['z'][1] - roi['z'][0]) / grid_size + 1))
        y_shape = int(math.ceil((roi['y'][1] - roi['y'][0]) / grid_size))
        x_shape = int(math.ceil((roi['x'][1] - roi['x'][0]) / grid_size + 1))
        
        spatial_shape = [z_shape, y_shape, x_shape]
        sparse_features, sparse_indices = data_dic['sparse_features'], data_dic['sparse_indices']
        sparse_features_pow = sparse_features[:, :4]
        sparse_features_dop = torch.cat((sparse_features[:, :3], sparse_features[:, 4:5]), dim = -1)

        input_sp_tensor = spconv.SparseConvTensor(
            features=sparse_features_pow,
            indices=sparse_indices.int(),
            spatial_shape=spatial_shape,
            batch_size=data_dic['batch_size']
        )

        input_sp_tensor_dop = spconv.SparseConvTensor(
            features=sparse_features_dop,
            indices=sparse_indices.int(),
            spatial_shape=spatial_shape,
            batch_size=data_dic['batch_size']
        )

        bev_1_pw, bev_2_pw, bev_3_pw = self.sham_pw(input_sp_tensor)
        bev_1_dop, bev_2_dop, bev_3_dop = self.sham_dop(input_sp_tensor_dop)
        
        bev_1 = self.convtrans2d_1(torch.cat((bev_1_pw, bev_1_dop), dim=1)) # B, C, X, Y
        bev_1 = self.bnt1(bev_1)
        bev_1 = self.relu(bev_1)

        bev_2 = self.convtrans2d_2(torch.cat((bev_2_pw, bev_2_dop), dim=1)) # B, C, X, Y
        bev_2 = self.bnt2(bev_2)
        bev_2 = self.relu(bev_2)

        bev_3 = self.convtrans2d_3(torch.cat((bev_3_pw, bev_3_dop), dim=1)) # B, C, X, Y
        bev_3 = self.bnt3(bev_3)
        bev_3 = self.relu(bev_3)

        bev_features = torch.cat((bev_1, bev_2, bev_3), dim = 1)
        data_dic['out_feat'] = bev_features
        
        return data_dic
