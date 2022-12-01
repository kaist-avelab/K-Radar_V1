"""
# -*- coding: utf-8 -*-
--------------------------------------------------------------------------------
# author: Donghee Paek, AVELab, KAIST
# date:   2022.01.05
# e-mail: donghee.paek@kaist.ac.kr
--------------------------------------------------------------------------------
# description: code for making head
"""

from .base_backbone_3d_sparse import BaseBackbone3DSparse
from .base_backbone_3d_sparse_dop_high_semantic import BaseBackbone3DSparseDopHSemantic

__all__ = {
    'BaseBackbone3DSparse': BaseBackbone3DSparse,
    'BaseBackbone3DSparseDopHSemantic': BaseBackbone3DSparseDopHSemantic
}
