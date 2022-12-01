'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import os
os.environ['CUDA_VISIBLE_DEVICES']= '0'

from pipelines.pipeline_v2_1 import Pipeline_v2_1

if __name__ == '__main__':
    pline = Pipeline_v2_1( \
        path_cfg = './configs/cfg_RTNH.yml', \
        split = 'train', \
        mode  = 'train/val')
    pline.train_network()
