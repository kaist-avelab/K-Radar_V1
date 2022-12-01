'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

from .pointpillar_backbone import PointPillarBackbone
from .resnet_wrapper import ResNetFPN
from .resnet_wrapper_multires import ResNetFpnMultiRes

__all__ = {
    'PointPillarBackbone': PointPillarBackbone,
    'ResNetFPN': ResNetFPN,
    'ResNetFpnMultiRes': ResNetFpnMultiRes,
}