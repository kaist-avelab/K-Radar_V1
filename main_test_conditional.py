'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from pipelines.pipeline_v2_1 import Pipeline_v2_1

if __name__ == '__main__':
    pline = Pipeline_v2_1('./configs/cfg_RTNH.yml', split='test', mode='train/val')
    pline.load_dict_model('./logs/RTNH/models/model_9.pt')
    pline.validate_kitti_conditional(list_conf_thr = [0.3, 0.5, 0.7], is_subset=False)
