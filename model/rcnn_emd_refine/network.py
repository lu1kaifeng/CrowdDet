import paddle as torch
#from torch import nn
from paddle import  nn
import paddle.nn.functional as F
import numpy as np
from paddle.fluid.layers import concat as cat


from .config import config
from backbone.resnet50 import ResNet50
from backbone.fpn import FPN
from module.rpn import RPN
from layers.pooler import roi_pooler
from det_oprs.bbox_opr import bbox_transform_inv_opr
from det_oprs.fpn_roi_target import fpn_roi_target
from det_oprs.loss_opr import emd_loss_softmax
from det_oprs.utils import get_padded_tensor


class Network(nn.Layer):
    def __init__(self):
        super().__init__()
        #self.resnet50 = ResNet50(config.backbone_freeze_at, False)
        self.resnet50 = ResNet50(config.backbone_freeze_at, False)
        self.FPN = FPN(self.resnet50, 2, 6)
        self.RPN = RPN(config.rpn_channel)
        self.RCNN = RCNN()
        assert config.num_classes == 2, 'Only support two class(1fg/1bg).'

    def forward(self, image, im_info, gt_boxes=None):
        image = (image - torch.cast(torch.to_tensor(config.image_mean[None, :, None, None]),'float32')) / (
                torch.cast(torch.to_tensor(config.image_std[None, :, None, None]),'float32'))
        image = get_padded_tensor(image, 64)
        if self.training:
            return self._forward_train(image, im_info, gt_boxes)
        else:
            return self._forward_test(image, im_info)

    def _forward_train(self, image, im_info, gt_boxes):
        loss_dict = {}
        fpn_fms = self.FPN(image)
        # fpn_fms stride: 64,32,16,8,4, p6->p2
        rpn_rois, loss_dict_rpn = self.RPN(fpn_fms, im_info, gt_boxes)
        rcnn_rois, rcnn_labels, rcnn_bbox_targets = fpn_roi_target(
                rpn_rois, im_info, gt_boxes, top_k=2)
        loss_dict_rcnn = self.RCNN(fpn_fms, rcnn_rois,
                rcnn_labels, rcnn_bbox_targets)
        loss_dict.update(loss_dict_rpn)
        loss_dict.update(loss_dict_rcnn)
        return loss_dict

    def _forward_test(self, image, im_info):
        fpn_fms = self.FPN(image)
        rpn_rois = self.RPN(fpn_fms, im_info)
        pred_bbox = self.RCNN(fpn_fms, rpn_rois)
        return pred_bbox.cpu().detach()

class RCNN(nn.Layer):
    def __init__(self):
        super().__init__()
        # roi head
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1044, 1024)

        # for l in [self.fc1, self.fc2, self.fc3]:
        #     nn.init.kaiming_uniform_(l.weight, a=1)
        #     nn.init.constant_(l.bias, 0)
        # box predictor
        self.emd_pred_cls_0 = nn.Linear(1024, config.num_classes)
        self.emd_pred_delta_0 = nn.Linear(1024, config.num_classes * 4)
        self.emd_pred_cls_1 = nn.Linear(1024, config.num_classes)
        self.emd_pred_delta_1 = nn.Linear(1024, config.num_classes * 4)
        self.ref_pred_cls_0 = nn.Linear(1024, config.num_classes)
        self.ref_pred_delta_0 = nn.Linear(1024, config.num_classes * 4)
        self.ref_pred_cls_1 = nn.Linear(1024, config.num_classes)
        self.ref_pred_delta_1 = nn.Linear(1024, config.num_classes * 4)
        # for l in [self.emd_pred_cls_0, self.emd_pred_cls_1,
        #         self.ref_pred_cls_0, self.ref_pred_cls_1]:
        #     nn.init.normal_(l.weight, std=0.001)
        #     nn.init.constant_(l.bias, 0)
        # for l in [self.emd_pred_delta_0, self.emd_pred_delta_1,
        #         self.ref_pred_delta_0, self.ref_pred_delta_1]:
        #     nn.init.normal_(l.weight, std=0.001)
        #     nn.init.constant_(l.bias, 0)

    def forward(self, fpn_fms, rcnn_rois, labels=None, bbox_targets=None):
        # stride: 64,32,16,8,4 -> 4, 8, 16, 32
        fpn_fms = fpn_fms[1:][::-1]
        stride = [4, 8, 16, 32]
        pool_features = roi_pooler(fpn_fms, rcnn_rois, stride, (7, 7), "ROIAlignV2")
        flatten_feature = torch.flatten(pool_features, start_axis=1)
        flatten_feature = F.relu_(self.fc1(flatten_feature))
        flatten_feature = F.relu_(self.fc2(flatten_feature))
        pred_emd_cls_0 = self.emd_pred_cls_0(flatten_feature)
        pred_emd_delta_0 = self.emd_pred_delta_0(flatten_feature)
        pred_emd_cls_1 = self.emd_pred_cls_1(flatten_feature)
        pred_emd_delta_1 = self.emd_pred_delta_1(flatten_feature)
        pred_emd_scores_0 = F.softmax(pred_emd_cls_0, axis=-1)
        pred_emd_scores_1 = F.softmax(pred_emd_cls_1, axis=-1)
        # cons refine feature
        boxes_feature_0 = cat((pred_emd_delta_0[:, 4:],
            pred_emd_scores_0[:, 1][:, None]), axis=1).repeat(1, 4)
        boxes_feature_1 = cat((pred_emd_delta_1[:, 4:],
            pred_emd_scores_1[:, 1][:, None]), axis=1).repeat(1, 4)
        boxes_feature_0 = cat((flatten_feature, boxes_feature_0), axis=1)
        boxes_feature_1 = cat((flatten_feature, boxes_feature_1), axis=1)
        refine_feature_0 = F.relu_(self.fc3(boxes_feature_0))
        refine_feature_1 = F.relu_(self.fc3(boxes_feature_1))
        # refine
        pred_ref_cls_0 = self.ref_pred_cls_0(refine_feature_0)
        pred_ref_delta_0 = self.ref_pred_delta_0(refine_feature_0)
        pred_ref_cls_1 = self.ref_pred_cls_1(refine_feature_1)
        pred_ref_delta_1 = self.ref_pred_delta_1(refine_feature_1)
        if self.training:
            loss0 = emd_loss_softmax(
                        pred_emd_delta_0, pred_emd_cls_0,
                        pred_emd_delta_1, pred_emd_cls_1,
                        bbox_targets, labels)
            loss1 = emd_loss_softmax(
                        pred_emd_delta_1, pred_emd_cls_1,
                        pred_emd_delta_0, pred_emd_cls_0,
                        bbox_targets, labels)
            loss2 = emd_loss_softmax(
                        pred_ref_delta_0, pred_ref_cls_0,
                        pred_ref_delta_1, pred_ref_cls_1,
                        bbox_targets, labels)
            loss3 = emd_loss_softmax(
                        pred_ref_delta_1, pred_ref_cls_1,
                        pred_ref_delta_0, pred_ref_cls_0,
                        bbox_targets, labels)
            loss_rcnn = cat([loss0, loss1], axis=1)
            loss_ref = cat([loss2, loss3], axis=1)
            # requires_grad = False
            _, min_indices_rcnn = loss_rcnn.min(dim=1)
            _, min_indices_ref = loss_ref.min(dim=1)
            loss_rcnn = loss_rcnn[torch.arange(loss_rcnn.shape[0]), min_indices_rcnn]
            loss_rcnn = loss_rcnn.mean()
            loss_ref = loss_ref[torch.arange(loss_ref.shape[0]), min_indices_ref]
            loss_ref = loss_ref.mean()
            loss_dict = {}
            loss_dict['loss_rcnn_emd'] = loss_rcnn
            loss_dict['loss_ref_emd'] = loss_ref
            return loss_dict
        else:
            class_num = pred_ref_cls_0.shape[-1] - 1
            tag = torch.arange(class_num).type_as(pred_ref_cls_0)+1
            tag = tag.repeat(pred_ref_cls_0.shape[0], 1).reshape(-1,1)
            pred_scores_0 = F.softmax(pred_ref_cls_0, axis=-1)[:, 1:].reshape(-1, 1)
            pred_scores_1 = F.softmax(pred_ref_cls_1, axis=-1)[:, 1:].reshape(-1, 1)
            pred_delta_0 = pred_ref_delta_0[:, 4:].reshape(-1, 4)
            pred_delta_1 = pred_ref_delta_1[:, 4:].reshape(-1, 4)
            base_rois = rcnn_rois[:, 1:5].repeat(1, class_num).reshape(-1, 4)
            pred_bbox_0 = restore_bbox(base_rois, pred_delta_0, True)
            pred_bbox_1 = restore_bbox(base_rois, pred_delta_1, True)
            pred_bbox_0 = cat([pred_bbox_0, pred_scores_0, tag], axis=1)
            pred_bbox_1 = cat([pred_bbox_1, pred_scores_1, tag], axis=1)
            pred_bbox = cat((pred_bbox_0, pred_bbox_1), axis=1)
            return pred_bbox

def restore_bbox(rois, deltas, unnormalize=True):
    if unnormalize:
        std_opr = torch.to_tensor(config.bbox_normalize_stds[None, :]).type_as(deltas)
        mean_opr = torch.to_tensor(config.bbox_normalize_means[None, :]).type_as(deltas)
        deltas = deltas * std_opr
        deltas = deltas + mean_opr
    pred_bbox = bbox_transform_inv_opr(rois, deltas)
    return pred_bbox
