import paddle as torch
from paddle import nn
import paddle.nn.functional as F
import numpy as np
from rcnn_emd_refine.config import config
from det_oprs.anchors_generator import AnchorGenerator
from det_oprs.find_top_rpn_proposals import find_top_rpn_proposals
from det_oprs.fpn_anchor_target import fpn_anchor_target, fpn_rpn_reshape
from det_oprs.loss_opr import softmax_loss as slo, smooth_l1_loss
from paddle.fluid.layers import softmax_with_cross_entropy as softmax_loss
class RPN(nn.Layer):
    def __init__(self, rpn_channel = 256):
        super().__init__()
        self.anchors_generator = AnchorGenerator(
            config.anchor_base_size,
            config.anchor_aspect_ratios,
            config.anchor_base_scale)
        self.rpn_conv = nn.Conv2D(256, rpn_channel, kernel_size=3, stride=1, padding=1)
        self.rpn_cls_score = nn.Conv2D(rpn_channel, config.num_cell_anchors * 2, kernel_size=1, stride=1)
        self.rpn_bbox_offsets = nn.Conv2D(rpn_channel, config.num_cell_anchors * 4, kernel_size=1, stride=1)

        # for l in [self.rpn_conv, self.rpn_cls_score, self.rpn_bbox_offsets]:
        #     nn.init.normal_(l.weight, std=0.01)
        #     nn.init.constant_(l.bias, 0)

    def forward(self, features, im_info, boxes=None):
        # prediction
        pred_cls_score_list = []
        pred_bbox_offsets_list = []
        for x in features:
            t = F.relu(self.rpn_conv(x))
            pred_cls_score_list.append(self.rpn_cls_score(t))
            pred_bbox_offsets_list.append(self.rpn_bbox_offsets(t))
        # get anchors
        all_anchors_list = []
        # stride: 64,32,16,8,4 p6->p2
        base_stride = 4
        off_stride = 2**(len(features)-1) # 16
        for fm in features:
            layer_anchors = self.anchors_generator(fm, base_stride, off_stride)
            off_stride = off_stride // 2
            all_anchors_list.append(layer_anchors)
        # sample from the predictions
        rpn_rois = find_top_rpn_proposals(
                self.training, pred_bbox_offsets_list, pred_cls_score_list,
                all_anchors_list, im_info)
        rpn_rois = rpn_rois.cast('float32')
        if self.training:
            rpn_labels, rpn_bbox_targets = fpn_anchor_target(
                    boxes, im_info, all_anchors_list)
            #rpn_labels = rpn_labels.astype(np.int32)
            pred_cls_score, pred_bbox_offsets = fpn_rpn_reshape(
                pred_cls_score_list, pred_bbox_offsets_list)
            # rpn loss
            valid_masks = rpn_labels >= 0
            # objectness_loss = softmax_loss(
            #     torch.gather(pred_cls_score,torch.nonzero(valid_masks)),
            #     torch.gather(rpn_labels,torch.nonzero(valid_masks)))

            objectness_loss = F.binary_cross_entropy(F.softmax(torch.gather(pred_cls_score, torch.nonzero(valid_masks))),
            torch.gather(torch.eye(2),torch.gather(rpn_labels, torch.nonzero(valid_masks))))

            pos_masks = rpn_labels > 0
            # localization_loss = smooth_l1_loss(
            #     pred_bbox_offsets[pos_masks],
            #     rpn_bbox_targets[pos_masks],
            #     config.rpn_smooth_l1_beta)
            localization_loss = \
            F.smooth_l1_loss(torch.gather(pred_bbox_offsets, torch.nonzero(pos_masks)),
                             torch.gather(rpn_bbox_targets, torch.nonzero(pos_masks)),delta=config.rcnn_smooth_l1_beta)
            normalizer = 1 / valid_masks.cast('float32').sum()
            loss_rpn_cls = objectness_loss.sum() * normalizer
            loss_rpn_loc = localization_loss.sum() * normalizer
            loss_dict = {}
            loss_dict['loss_rpn_cls'] = loss_rpn_cls
            loss_dict['loss_rpn_loc'] = loss_rpn_loc
            return rpn_rois, loss_dict
        else:
            return rpn_rois

