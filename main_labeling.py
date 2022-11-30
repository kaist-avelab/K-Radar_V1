"""
# -*- coding: utf-8 -*-
--------------------------------------------------------------------------------
# author: Donghee Paek, AVELab, KAIST
# date:   2021.10.07
# e-mail: donghee.paek@kaist.ac.kr
--------------------------------------------------------------------------------
# description: script for object detection labeling
"""

import configs.config_general as cnf
from uis.ui_labeling import startUi

if __name__ == '__main__':
    print(cnf.BASE_DIR)
    startUi()
    