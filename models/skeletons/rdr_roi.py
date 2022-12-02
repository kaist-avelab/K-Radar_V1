'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import torch.nn as nn

from models import middle_encoder, backbone_2d, backbone_3d, head

class RadarRoI(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_model = cfg.MODEL
        
        self.list_module_names = [
            'middle_encoder', 'backbone', 'head', 'roi_head'
        ]
        self.list_modules = []
        self.build_radar_detector()

    def build_radar_detector(self):
        for name_module in self.list_module_names:
            module = getattr(self, f'build_{name_module}')()
            if module is not None:
                self.add_module(name_module, module) # override nn.Module
                self.list_modules.append(module)

    def build_middle_encoder(self):
        if self.cfg_model.get('MIDDLE_ENCODER', None) is None:
            return None
        
        module = middle_encoder.__all__[self.cfg_model.MIDDLE_ENCODER.NAME](self.cfg)
        return module

    def build_backbone(self):
        cfg_backbone = self.cfg_model.get('BACKBONE', None)
        if cfg_backbone is None:
            return None
        
        if cfg_backbone.TYPE == '2D':
            return backbone_2d.__all__[cfg_backbone.NAME](self.cfg)
        elif cfg_backbone.TYPE == '3D':
            return backbone_3d.__all__[cfg_backbone.NAME](self.cfg)
        else:
            return None

    def build_head(self):
        if (self.cfg.MODEL.get('HEAD', None)) is None:
            return None
        module = head.__all__[self.cfg_model.HEAD.NAME](self.cfg)
        return module

    def build_roi_head(self):
        if (self.cfg.MODEL.get('ROI_HEAD', None)) is None:
            return None
        head_module = roi_head.__all__[self.cfg_model.ROI_HEAD.NAME](self.cfg)
        return head_module

    def forward(self, x):
        for module in self.list_modules:
            x = module(x)
        
        return x
