'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import torch
import torch.nn as nn
import numpy as np
import nms

from utils.Rotated_IoU.oriented_iou_loss import cal_iou, cal_iou_3d
 
class FocalLoss(nn.Module):
    def __init__(self, weight=None, 
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor):
        log_prob = nn.functional.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)

        return nn.functional.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )

class PointPillarHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg


        self.anchors_per_location = []
        num_anchors = 0

        if (self.cfg.DATASET.BOX_CODE[-1] == 'cos') or (self.cfg.DATASET.BOX_CODE[-1] == 'sin'):
            self.USE_SIN_COS = True
        else:
            self.USE_SIN_COS = False

        for anchor in self.cfg.MODEL.HEAD.ANCHOR_GENERATOR_CONFIG:
            self.anchor_sizes = (anchor['anchor_sizes'])
            self.anchor_rotations = (anchor['anchor_rotations'])
            self.anchor_bottoms = (anchor['anchor_bottom_heights'])
            
            for anchor_size in self.anchor_sizes:
                for anchor_rot in self.anchor_rotations:
                    for anchor_bottom in self.anchor_bottoms:
                        temp_anchor = [anchor_bottom]
                        temp_anchor += anchor_size
                        temp_anchor += [np.cos(anchor_rot)]
                        temp_anchor += [np.sin(anchor_rot)]
                        num_anchors += 1
                        self.anchors_per_location.append(temp_anchor)

        self.num_anchors_per_location = num_anchors
        self.num_class = self.cfg.DATASET.NUM_CLS
        # self.num_anchors_per_class = num_anchors / (self.num_class - 1)
        self.box_code_size = len(self.cfg.DATASET.BOX_CODE)

        input_channels = sum(self.cfg.MODEL.BACKBONE_2D.NUM_UPSAMPLE_FILTERS)
        
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location + 1,  # plus one for background
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_code_size,
            kernel_size=1
        )

        # if self.cfg.MODEL.HEAD.USE_DIRECTION_CLASSIFIER:
        #     self.conv_dir_cls = nn.Conv2d(
        #         input_channels,
        #         self.num_anchors_per_location * self.cfg.MODEL.HEAD.NUM_DIR_BINS,
        #         kernel_size=1
        #     )
        # else:
        #     self.conv_dir_cls = None
        # self.init_weights()
        self.categorical_focal_loss = FocalLoss()
        self.is_logging = cfg.GENERAL.LOGGING.IS_LOGGING

    def forward(self, data_dic):
        spatial_features_2d = data_dic['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d) # B x Num_Anc+1 * 1 x W x H
        box_preds = self.conv_box(spatial_features_2d) # B x Num_Anc * Box_Size x W x H

        preds_dic = {
            'cls_preds': cls_preds,
            'box_preds': box_preds,
            'dir_preds': None
        }

        # if self.cfg.MODEL.HEAD.USE_DIRECTION_CLASSIFIER:
        #     dir_preds = self.conv_dir_cls(spatial_features_2d) # B x Num_Anc * C x W x H
        #     preds_dic['dir_preds'] = dir_preds
        
        data_dic['preds'] = preds_dic
        data_dic['anchor_maps'] = self.create_anchors(data_dic)
        # data_dic['loss'] = self.loss(data_dic)

        return data_dic

    def loss(self, data_dic):
        # import time
        # start_time = time.time()

        dtype, device = data_dic['preds']['cls_preds'].dtype, data_dic['preds']['cls_preds'].device 

        anchor_maps = data_dic['anchor_maps'] # B x Num_Anc * C x W x H --> xc,yc,zc,...,cos,sin,xc,yc,...
        cls_preds = data_dic['preds']['cls_preds'] # B x Num_Anc * C x W x H
        box_preds = data_dic['preds']['box_preds'] # B x Num_Anc * C x W x H

        box_preds = anchor_maps + box_preds # change prediction into residuals
        
        # if self.cfg.MODEL.HEAD.USE_DIRECTION_CLASSIFIER:
        #     dir_preds = data_dic['preds']['dir_preds']

        # print('-----')
        # print('>>>', cls_preds.shape)
        # print('<<<', box_preds.shape)
        # raise ValueError
        B, _, W, H = cls_preds.shape

        cls_preds = cls_preds.view(B, self.num_anchors_per_location+1, W, H) # B x Num_Anc+1 x W x H
        box_preds = box_preds.view(B, self.num_anchors_per_location, -1, W, H) # B x Num_Anc x Box_Size x W x H    

        # Figure out which anchors are positives
        dtype, device = cls_preds.dtype, cls_preds.device
        
        cls_targets = torch.full((B, W, H), -1, dtype = dtype, device = device)
        pos_box_preds = []
        pos_box_targets = []

        is_label_valid = False

        for batch_id, batch_labels in enumerate(data_dic['labels']):
            # start_batch_iter = time.time()

            filtered_labels = []
            for bbox in batch_labels:
                # _, _, [x, y, z, theta, xl, yl, zl], _ = bbox
                # obj3d = Object3D(x, y, z, xl, yl, zl, theta)


                # pts = [obj3d.corners[0,:], obj3d.corners[2,:], obj3d.corners[4,:], obj3d.corners[6,:]]
                is_bbox_valid = True
                # rdr_cube_bev = data_dic['rdr_cube_bev']
                # for pt in pts:
                #     idx_x = np.argmin(np.abs(self.arr_x_cb-pt[0]))
                #     idx_y = np.argmin(np.abs(self.arr_y_cb-pt[1]))
                #     # print(rdr_cube_bev.shape)
                #     # print(idx_x, idx_y)
                #     # print(rdr_cube_bev[batch_id,0,idx_y,idx_x])
                #     if rdr_cube_bev[batch_id,0,idx_y,idx_x] == 0.:
                #         is_bbox_valid=False
                if is_bbox_valid:
                    filtered_labels.append(bbox)
                    is_label_valid = True

            anchors_cars = torch.cat((box_preds[batch_id, :2, :2], box_preds[batch_id, :2, 3:5], torch.atan(box_preds[batch_id, :2, 6:7] / box_preds[batch_id, :2, 5:6])), dim = 1) #BEV
            anchors_cars = anchors_cars.permute(0, 2, 3, 1).contiguous() # Num_Anc x W x H x 5
            anchors_cars = anchors_cars.view(1,-1,5)

            # anchors_buss = torch.cat((box_preds[batch_id, 2:4, :2], box_preds[batch_id, 2:4, 3:5], torch.atan(box_preds[batch_id, 2:4, 6:7] / box_preds[batch_id, 2:4, 5:6])), dim = 1) #BEV
            # anchors_buss = anchors_buss.permute(0, 2, 3, 1).contiguous() # Num_Anc x W x H x 5
            # anchors_buss = anchors_buss.view(1,-1,5)

            # anchors_peds = torch.cat((box_preds[batch_id, 2:4, :2], box_preds[batch_id, 4:6, 3:5], torch.atan(box_preds[batch_id, 4:6, 6:7] / box_preds[batch_id, 4:6, 5:6])), dim = 1) #BEV
            # anchors_peds = anchors_peds.permute(0, 2, 3, 1).contiguous() # Num_Anc x W x H x 5
            # anchors_peds = anchors_peds.view(1,-1,5)

            # anchors_cycs = torch.cat((box_preds[batch_id, 4:6, :2], box_preds[batch_id, 6:8, 3:5], torch.atan(box_preds[batch_id, 6:8, 6:7] / box_preds[batch_id, 6:8, 5:6])), dim = 1) #BEV
            # anchors_cycs = anchors_cycs.permute(0, 2, 3, 1).contiguous() # Num_Anc x W x H x 5
            # anchors_cycs = anchors_cycs.view(1,-1,5)

            num_locations_per_anchor = anchors_cars.shape[1] // 2

            # label_tensor = torch.zeros((len(batch_labels), 5), dtype = cls_preds.dtype, device = cls_preds.device)
            # print('>>>>>>>>>>>>>>>>', len(batch_labels))

            for label_id, label in enumerate(batch_labels):
                # start_label_iter = time.time()
                
                _, cls_id, (xc, yc, zc, rz, xl, yl, zl), _ = label

                if (len([xc, yc, xl, yl, rz]) > 0):
                    # label_tensor[label_id] = torch.tensor([xc, yc, xl, yl, rz], dtype = dtype, device = device)
                    if (cls_id == 1):
                        anchors = anchors_cars
                        cls_target_id = 1 # 1 or 2 for cars
                    # elif (cls_id == 2):
                    #     anchors = anchors_buss
                    #     cls_target_id = 3 # 3 or 4 for buses
                    # elif (cls_id == 3):
                    #     anchors = anchors_peds
                    #     cls_target_id = 5 # 5 or 6 for peds
                    # elif (cls_id == 4):
                    #     anchors = anchors_cycs
                    #     cls_target_id = 7 # 5 or 6 for cyclists
                    else:
                        anchors = None
                        cls_target_id = 0 # 0 for background

                    # print(anchors)
                    if (anchors != None):
                        label_tensor = torch.tensor([xc, yc, xl, yl, rz], dtype = dtype, device = device)
                        label_tensor = label_tensor.unsqueeze(0).unsqueeze(0).repeat(1, anchors.shape[1], 1)  
                        iou, _, _, _ = cal_iou(label_tensor, anchors)
                    
                    pos_ious_ind = torch.where(iou > 0.5)[1] # anchor idx
                    # at least 1 box
                    if len(pos_ious_ind)==0:
                        pos_ious_ind = [torch.argmax(iou)]

                    # print(f'idx = {pos_ious_ind}')
                    
                    neg_ious_ind  = torch.where(iou < 0.2)[1] # anchor idx

                    # Making negative targets (Much much faster than looping)
                    neg_ious_ind = torch.remainder(neg_ious_ind, num_locations_per_anchor)
                    W_targets, H_targets = torch.div(neg_ious_ind, H, rounding_mode='trunc'), torch.remainder(neg_ious_ind, H)
                    cls_targets[batch_id, W_targets, H_targets] = 0

                    # Making positive targets
                    for anchor_idx in pos_ious_ind:
                        # print('>>>',anchor_idx)
                        temp_id = cls_target_id
                        if (anchor_idx > num_locations_per_anchor):
                            temp_id = cls_target_id + 1
                        
                        anchor_idx = anchor_idx % num_locations_per_anchor
                        W_target, H_target = torch.div(anchor_idx, H, rounding_mode='trunc'), torch.remainder(anchor_idx, H)
                        cls_targets[batch_id, W_target, H_target] = temp_id
                        pos_box_preds.append(box_preds[batch_id:batch_id+1, temp_id-1, :, W_target, H_target])
                        pos_box_targets.append(torch.tensor([[xc, yc, zc, xl, yl, zl, np.cos(rz).item(), np.sin(rz).item()]], dtype = dtype, device = device))
                    
        # start_loss_calc = time.time()

        # print('-'*30)
        # print(pos_box_preds)
        # print('='*30)
        
        if not is_label_valid: # two scenes without labels
            loss_reg = 0.
            focal_loss_cls = 0.
        else:
            counted_cls_idx = torch.where(cls_targets > -1)
            cls_targets_counted = cls_targets[counted_cls_idx] # pos and neg boxes only
            cls_preds_counted = cls_preds[counted_cls_idx[0], :, counted_cls_idx[1], counted_cls_idx[2]]
            cls_targets[torch.where(cls_targets == -1)] = 0

            cls_weights = torch.ones(self.num_anchors_per_location + 1, device = device)

            pos_box_preds = torch.cat(pos_box_preds)
            pos_box_targets = torch.cat(pos_box_targets)
            loss_reg = torch.nn.functional.smooth_l1_loss(pos_box_preds, pos_box_targets)
            cls_weights[0] = min(10/neg_ious_ind.shape[0], 1) # weights for background class
        
            self.categorical_focal_loss.weight = cls_weights
            focal_loss_cls = self.categorical_focal_loss(cls_preds_counted, cls_targets_counted.long())

        total_loss = focal_loss_cls + loss_reg

        if self.is_logging:
            data_dic['log_loss'] = dict()
            data_dic['log_loss'].update(self.logging_dict_loss(total_loss, 'total_loss'))
            data_dic['log_loss'].update(self.logging_dict_loss(loss_reg, 'loss_reg'))
            data_dic['log_loss'].update(self.logging_dict_loss(focal_loss_cls, 'focal_loss_cls'))

        return total_loss

    def logging_dict_loss(self, loss, name_key):
        try:
            log_loss = loss.cpu().detach().item()
        except:
            log_loss = loss
        
        return {name_key: log_loss}

    def create_anchors(self, data_dic):
        '''
            If we have two anchors (1,2) per class for three classes (A,B,C), the order will be A1 A2 B1 B2 C1 C2
        '''
        x_min, y_min, z_min, x_max, y_max, z_max = self.cfg.MODEL.VOXEL_ENCODER.LDR_PC_RANGE
        B, _, y_grid_range, x_grid_range = data_dic['preds']['cls_preds'].shape
        dtype, device = data_dic['preds']['cls_preds'].dtype, data_dic['preds']['cls_preds'].device
        x_grid_size, y_grid_size = (x_max - x_min) / x_grid_range, (y_max - y_min) / y_grid_range

        anchor_x = torch.arange(x_min, x_max, x_grid_size, dtype=dtype, device=device) + x_grid_size/2 # anchor location is in the middle of the grid
        anchor_y = torch.arange(y_min, y_max, y_grid_size, dtype=dtype, device=device) + y_grid_size/2 # anchor location is in the middle of the grid

        anchor_y = anchor_y.repeat_interleave(x_grid_range)
        anchor_x = anchor_x.repeat(y_grid_range)

        flat_anchor_map = torch.stack((anchor_x, anchor_y), dim = 1).unsqueeze(0).repeat(self.num_anchors_per_location, 1, 1) # Num_Anc x H * W x 2
        flat_anc_attr = torch.tensor(self.anchors_per_location, 
                            dtype = flat_anchor_map.dtype, 
                            device = flat_anchor_map.device).unsqueeze(1).repeat(1, flat_anchor_map.shape[1], 1)
        anchor_map = torch.cat((flat_anchor_map, flat_anc_attr), dim = -1).view(self.num_anchors_per_location, y_grid_range, x_grid_range, -1) # Num_Anc x W x H x Attr(8)
        batch_anchor_map = anchor_map.unsqueeze(0).repeat(B, 1, 1, 1, 1).contiguous().permute(0,1,4,2,3) # B x Num_Anc x C x W x H
        batch_anchor_map = batch_anchor_map.reshape(B, -1, y_grid_range, x_grid_range).contiguous() # B x Num_Anc * C x W x H --> xc,yc,zc,...,cos,sin,xc,yc,...
        # print(batch_anchor_map[0, 0, :, 0, 0])
        # print(batch_anchor_map.shape)

        return batch_anchor_map

    def get_pred_boxes_nms_for_single_datum(self, dict_out, conf_thr):
        '''
        * Assume batch size = 1
        '''
        try:
            anchors = self.create_anchors(dict_out)[0]
            cls_preds, box_preds = dict_out['preds']['cls_preds'][0], dict_out['preds']['box_preds'][0]

            cls_preds, box_preds, anchors = cls_preds.view(cls_preds.shape[0], -1), box_preds.view(box_preds.shape[0], -1), anchors.view(anchors.shape[0], -1)
            cared_idx = torch.where((torch.argmax(cls_preds, dim = 0) > 0) & (torch.max(torch.softmax(cls_preds, dim = 0), dim=0)[0] > conf_thr))
            cls_preds = torch.softmax(cls_preds, dim = 0)

            cared_cls_preds, cared_box_preds, cared_anchors = cls_preds[:, cared_idx[0]],  box_preds[:, cared_idx[0]], anchors[:, cared_idx[0]] # Remove background predictions
            cared_cls_preds_id = torch.argmax(cared_cls_preds, dim = 0)

            cared_boxes = []
            cared_boxes_with_scores = []
            cls_ids = []
            for i, cls_id in enumerate(cared_cls_preds_id):
                score = cared_cls_preds[cls_id][i:i+1]
                start_id = (cls_id - 1)*8 # len(self.cfg.DATASET.BOX_CODE)
                residuals = cared_box_preds[start_id:start_id+8, i]
                anchor = cared_anchors[start_id:start_id+8, i]
                pred_cos_sin = residuals + anchor

                ### Check atan2 or atan ###
                angle = torch.atan2(pred_cos_sin[-1], pred_cos_sin[-2]).unsqueeze(0)
                pred = torch.concat((pred_cos_sin[:-2], angle))
                cared_boxes.append(pred)
        
                pred_with_scores = torch.concat((score, pred_cos_sin[:-2], angle))
                cared_boxes_with_scores.append(pred_with_scores)
                cls_ids.append(cls_id)

            if len(cared_boxes) == 0:
                # empty prediction
                pass
            else:
                cared_boxes = torch.stack(cared_boxes)
                cared_boxes_with_scores = torch.stack(cared_boxes_with_scores) # N_proposals x 8 (scores, xc, yc, zc, xl, yl, zl, angle)
                cls_ids = torch.stack(cls_ids)

                scores = cared_boxes_with_scores[:, 0:1].cpu().detach().numpy()
                xc_tensor, yc_tensor = cared_boxes_with_scores[:, 1:2], cared_boxes_with_scores[:, 2:3]
                xl_tensor, yl_tensor = cared_boxes_with_scores[:, 4:5], cared_boxes_with_scores[:, 5:6]
                angle_tensor = cared_boxes_with_scores[:, 7:8] 

                c_array = torch.cat((xc_tensor, yc_tensor), dim = 1).cpu().detach().numpy()
                dim_array = torch.cat((xl_tensor, yl_tensor), dim = 1).cpu().detach().numpy()
                angle_array = angle_tensor.cpu().detach().numpy()

                c_list = list(map(tuple, c_array))
                dim_list = list(map(tuple, dim_array))
                angle_list = list(map(float, angle_array))

                boxes = [[a, b , c] for (a, b, c) in zip(c_list, dim_list, angle_list)]

                nms_overlap_thresh = 0.3
                ### If NMS error, do not calculate nms ###
                try:
                    keep_indices = nms.rboxes(boxes, scores, nms_threshold=nms_overlap_thresh)
                    cared_boxes_with_scores = cared_boxes_with_scores[keep_indices]
                    cls_ids = cls_ids[keep_indices]
                    dict_out['desc'][0].update({'is_nms': True})
                except:
                    dict_out['desc'][0].update({'is_nms': False})
                    pass
                ### If NMS error, do not calculate nms ###

                dict_out['pred_boxes_nms'] = cared_boxes_with_scores
                dict_out['pred_cls_ids'] = cls_ids
                dict_out['pred_desc'] = dict_out['desc'][0]

            return dict_out
        
        except:
            # dict_out[]
            print('* Error happens in ')
            
            return None
