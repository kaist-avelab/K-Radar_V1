'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

from .mean_vfe import MeanVoxelEncoder
from .mean_vfe_lidar import MeanVoxelEncoderLidar
from .mean_vfe_radar import MeanVoxelEncoder_Radar
from .mean_vfe_radar_withDop import MeanVoxelEncoder_Radar_withDop

__all__ = {
    'MeanVoxelEncoder': MeanVoxelEncoder,
    'MeanVoxelEncoderLidar': MeanVoxelEncoderLidar,
    'MeanVoxelEncoder_Radar': MeanVoxelEncoder_Radar,
    'MeanVoxelEncoder_Radar_withDop': MeanVoxelEncoder_Radar_withDop
}