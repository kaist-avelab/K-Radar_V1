import kitti_common as kitti
from eval import get_official_eval_result
import numpy as np
def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]
# path_header = '/home/donghee/Desktop/KRadar/KRadar_0316/logs/RdrCubeBev_5_17_13_24_38/val_kitti/temp'
# path_header = '/home/donghee/Desktop/KRadar/KRadar_0316/logs/Rdr4DNet_5_19_15_45_27/val_kitti/ep_0/0.3'
path_header = "/home/donghee/Desktop/KRadar/KRadar_0316/logs/temp1/"
det_path = path_header+"/preds"
dt_annos = kitti.get_label_annos(det_path)
# print(dt_annos)
# print(len(dt_annos))
# dt_annos_new = []
# for dt_anno in dt_annos:
#     print(dt_anno)
#     dt_anno['score'] = np.array([1.0])
#     print(dt_anno)
# gt_path = "/home/donghee/Desktop/KRadar/KRadar_0316/logs/RdrCubeBev_5_18_13_52_48/val_kitti/temp/gts"
gt_path = path_header+"/gts"
# gt_split_file = "/home/donghee/Desktop/KRadar/KRadar_0316/logs/RdrCubeBev_5_18_13_52_48/val_kitti/temp/val.txt" # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
gt_split_file = path_header+"/val.txt" # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
val_image_ids = _read_imageset_file(gt_split_file)
gt_annos = kitti.get_label_annos(gt_path, val_image_ids)
# print(get_official_eval_result(gt_annos, dt_annos, 0)) # 6s in my computer

# eval_sed = get_official_eval_result(gt_annos, dt_annos, 0)
# print(eval_sed)
# print(type(eval_sed))

dict_metrics, eval_sed = get_official_eval_result(gt_annos, dt_annos, 0, iou_mode='hard', is_return_with_dict=True)

print(eval_sed)
print(dict_metrics)

# import kitti_common as kitti
# from eval import get_official_eval_result
# import numpy as np
# def _read_imageset_file(path):
#     with open(path, 'r') as f:
#         lines = f.readlines()
#     return [int(line) for line in lines]
# # det_path = "/home/donghee/Desktop/KRadar/KRadar_0316/logs/RdrCubeBev_5_17_13_24_38/val_kitti/temp/preds"
# det_path = "/home/donghee/Desktop/KRadar/KRadar_0316/logs/temp1/preds"
# dt_annos = kitti.get_label_annos(det_path)
# # print(dt_annos)
# # print(len(dt_annos))
# # dt_annos_new = []
# # for dt_anno in dt_annos:
# #     print(dt_anno)
# #     dt_anno['score'] = np.array([1.0])
# #     print(dt_anno)
# # gt_path = "/home/donghee/Desktop/KRadar/KRadar_0316/logs/RdrCubeBev_5_18_13_52_48/val_kitti/temp/gts"
# gt_path = "/home/donghee/Desktop/KRadar/KRadar_0316/logs/temp1/gts"
# # gt_split_file = "/home/donghee/Desktop/KRadar/KRadar_0316/logs/RdrCubeBev_5_18_13_52_48/val_kitti/temp/val.txt" # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
# gt_split_file = "/home/donghee/Desktop/KRadar/KRadar_0316/logs/temp1/val.txt" # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
# val_image_ids = _read_imageset_file(gt_split_file)
# gt_annos = kitti.get_label_annos(gt_path, val_image_ids)
# print(get_official_eval_result(gt_annos, dt_annos, 0)) # 6s in my computer
# # print(get_coco_eval_result(gt_annos, dt_annos, 0)) # 18s in my computer
