import paddle as torch
import numpy as np

from det_oprs.bbox_opr import box_overlap_opr, bbox_transform_opr
from rcnn_emd_refine.config import config
from paddle.fluid.layers import concat as cat, greater_than as gt,elementwise_mul as mul,fill_constant


def fpn_rpn_reshape(pred_cls_score_list, pred_bbox_offsets_list):
    final_pred_bbox_offsets_list = []
    final_pred_cls_score_list = []
    for bid in range(config.train_batch_per_gpu):
        batch_pred_bbox_offsets_list = []
        batch_pred_cls_score_list = []
        for i in range(len(pred_cls_score_list)):
            pred_cls_score_perlvl = pred_cls_score_list[i][bid] \
                .transpose((1, 2, 0)).reshape((-1, 2))
            pred_bbox_offsets_perlvl = pred_bbox_offsets_list[i][bid] \
                .transpose((1, 2, 0)).reshape((-1, 4))
            batch_pred_cls_score_list.append(pred_cls_score_perlvl)
            batch_pred_bbox_offsets_list.append(pred_bbox_offsets_perlvl)
        batch_pred_cls_score = cat(batch_pred_cls_score_list, axis=0)
        batch_pred_bbox_offsets = cat(batch_pred_bbox_offsets_list, axis=0)
        final_pred_cls_score_list.append(batch_pred_cls_score)
        final_pred_bbox_offsets_list.append(batch_pred_bbox_offsets)
    final_pred_cls_score = cat(final_pred_cls_score_list, axis=0)
    final_pred_bbox_offsets = cat(final_pred_bbox_offsets_list, axis=0)
    return final_pred_cls_score, final_pred_bbox_offsets


def fpn_anchor_target_opr_core_impl(
        gt_boxes, im_info, anchors, allow_low_quality_matches=True):
    ignore_label = config.ignore_label
    # get the gt boxes
    valid_gt_boxes = gt_boxes[:int(im_info[5]), :]
    valid_gt_boxes = torch.gather(valid_gt_boxes, torch.nonzero(gt(valid_gt_boxes[:, -1], torch.zeros(valid_gt_boxes[:, -1].shape))))
    # compute the iou matrix
    anchors = anchors.cast('float32')
    overlaps = box_overlap_opr(anchors, valid_gt_boxes[:, :4])
    # match the dtboxes
    max_overlaps = torch.max(overlaps, axis=1)
    argmax_overlaps = torch.argmax(overlaps, axis=1)
    # _, gt_argmax_overlaps = torch.max(overlaps, axis=0)
    gt_argmax_overlaps = my_gt_argmax(overlaps)
    del overlaps
    # all ignore
    labels = torch.ones(torch.to_tensor(anchors.shape[0])).cast('long') * ignore_label
    # set negative ones
    labels = labels * (max_overlaps >= config.rpn_negative_overlap)
    # set positive ones
    fg_mask = (max_overlaps >= config.rpn_positive_overlap)
    if allow_low_quality_matches:
        gt_id = torch.arange(valid_gt_boxes.shape[0]).cast('float32')
        #argmax_overlaps[gt_argmax_overlaps] = gt_id
        for i,j in zip(gt_argmax_overlaps,range(gt_argmax_overlaps.shape[0])):
            argmax_overlaps[i]  = gt_id[j]
            max_overlaps[i] = 1
        #max_overlaps[gt_argmax_overlaps] = 1
        fg_mask = (max_overlaps >= config.rpn_positive_overlap)
    # set positive ones
    fg_mask_ind = torch.nonzero(fg_mask, as_tuple=False).flatten()
    #labels[fg_mask_ind] = 1
    for i in fg_mask_ind:
        labels[i] = 1
    # bbox targets
    bbox_targets = bbox_transform_opr(
        anchors, torch.gather(valid_gt_boxes,argmax_overlaps)[:, :4])
    if config.rpn_bbox_normalize_targets:
        std_opr = torch.to_tensor(config.bbox_normalize_stds[None, :]).type_as(bbox_targets)
        mean_opr = torch.to_tensor(config.bbox_normalize_means[None, :]).type_as(bbox_targets)
        minus_opr = mean_opr / std_opr
        bbox_targets = bbox_targets / std_opr - minus_opr
    return labels, bbox_targets


@torch.no_grad()
def fpn_anchor_target(boxes, im_info, all_anchors_list):
    final_labels_list = []
    final_bbox_targets_list = []
    for bid in range(config.train_batch_per_gpu):
        batch_labels_list = []
        batch_bbox_targets_list = []
        for i in range(len(all_anchors_list)):
            anchors_perlvl = all_anchors_list[i]
            rpn_labels_perlvl, rpn_bbox_targets_perlvl = fpn_anchor_target_opr_core_impl(
                boxes[bid], im_info[bid], anchors_perlvl)
            batch_labels_list.append(rpn_labels_perlvl)
            batch_bbox_targets_list.append(rpn_bbox_targets_perlvl)
        # here we samples the rpn_labels
        concated_batch_labels = cat(batch_labels_list, axis=0)
        concated_batch_bbox_targets = cat(batch_bbox_targets_list, axis=0)
        # sample labels
        pos_idx, neg_idx = subsample_labels(concated_batch_labels,
                                            config.num_sample_anchors, config.positive_anchor_ratio)
        #concated_batch_labels.fill_(-1)
        concated_batch_labels = fill_constant(concated_batch_labels.shape,'int32',-1)
        for p in pos_idx:
            concated_batch_labels[p] = 1
        for n in neg_idx:
            concated_batch_labels[n] = 0

        final_labels_list.append(concated_batch_labels)
        final_bbox_targets_list.append(concated_batch_bbox_targets)
    final_labels = cat(final_labels_list, axis=0)
    final_bbox_targets = cat(final_bbox_targets_list, axis=0)
    return final_labels, final_bbox_targets


def my_gt_argmax(overlaps):
    gt_max_overlaps = torch.max(overlaps, axis=0)
    gt_max_mask = overlaps == gt_max_overlaps
    gt_argmax_overlaps = []
    for i in range(overlaps.shape[-1]):
        gt_max_inds = torch.nonzero(gt_max_mask.cast('int')[:, i], as_tuple=False).flatten()
        gt_max_ind = torch.gather(gt_max_inds,torch.randperm(gt_max_inds.numel())[0])
        gt_argmax_overlaps.append(gt_max_ind)
    gt_argmax_overlaps = cat(gt_argmax_overlaps)
    return gt_argmax_overlaps


def subsample_labels(labels, num_samples, positive_fraction):
    positive = torch.nonzero(mul((labels != config.ignore_label).cast('int'),(labels != 0).cast('int')).cast('bool'), as_tuple=False).squeeze(1)
    negative = torch.nonzero(labels == 0, as_tuple=False).squeeze(1)

    num_pos = int(num_samples * positive_fraction)
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    if type(num_pos) == torch.Tensor:
        num_pos = num_pos.numpy().item()
    if type(num_neg) == torch.Tensor:
        num_neg = num_neg.numpy().item()
    perm1 = torch.randperm(positive.numel())[:num_pos]
    perm2 = torch.randperm(negative.numel())[:num_neg]

    pos_idx = torch.gather(positive,perm1)
    neg_idx = torch.gather(negative,perm2)
    return pos_idx, neg_idx
