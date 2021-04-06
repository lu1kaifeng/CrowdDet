import paddle as torch
from paddle.fluid.layers import concat as cat, locality_aware_nms as nms
from rcnn_emd_refine.config import config
from det_oprs.bbox_opr import bbox_transform_inv_opr, clip_boxes_opr, \
    filter_boxes_opr
import paddle.nn.functional as F
from torchvision.ops import nms as nmss
@torch.no_grad()
def find_top_rpn_proposals(is_train, rpn_bbox_offsets_list, rpn_cls_prob_list,
        all_anchors_list, im_info):
    prev_nms_top_n = config.train_prev_nms_top_n \
        if is_train else config.test_prev_nms_top_n
    post_nms_top_n = config.train_post_nms_top_n \
        if is_train else config.test_post_nms_top_n
    batch_per_gpu = config.train_batch_per_gpu if is_train else 1
    nms_threshold = config.rpn_nms_threshold
    box_min_size = config.rpn_min_box_size
    bbox_normalize_targets = config.rpn_bbox_normalize_targets
    bbox_normalize_means = config.bbox_normalize_means
    bbox_normalize_stds = config.bbox_normalize_stds
    list_size = len(rpn_bbox_offsets_list)

    return_rois = []
    return_inds = []
    for bid in range(batch_per_gpu):
        batch_proposals_list = []
        batch_probs_list = []
        for l in range(list_size):
            # get proposals and probs
            offsets = rpn_bbox_offsets_list[l][bid] \
                .transpose((1, 2, 0)).reshape((-1, 4))
            if bbox_normalize_targets:
                std_opr = torch.to_tensor(config.bbox_normalize_stds[None, :]).cast('float32')
                mean_opr = torch.to_tensor(config.bbox_normalize_means[None, :]).cast('float32')
                pred_offsets = pred_offsets * std_opr
                pred_offsets = pred_offsets + mean_opr
            all_anchors = all_anchors_list[l]
            proposals = bbox_transform_inv_opr(all_anchors, offsets)
            if config.anchor_within_border:
                proposals = clip_boxes_opr(proposals, im_info[bid, :])
            probs = rpn_cls_prob_list[l][bid] \
                    .transpose((1,2,0)).reshape((-1, 2))
            probs = F.softmax(probs, axis=-1)[:, 1]
            # gather the proposals and probs
            batch_proposals_list.append(proposals)
            batch_probs_list.append(probs)
        batch_proposals = cat(batch_proposals_list, axis=0)
        batch_probs = cat(batch_probs_list, axis=0)
        # filter the zero boxes.
        batch_keep_mask = filter_boxes_opr(
                batch_proposals, box_min_size * im_info[bid, 2])
        batch_keep_mask = torch.nonzero(batch_keep_mask)
        batch_proposals = torch.gather(batch_proposals,batch_keep_mask)
        batch_probs = torch.gather(batch_probs,batch_keep_mask)
        # prev_nms_top_n
        num_proposals = min(prev_nms_top_n, batch_probs.shape[0])
        batch_probs = batch_probs.sort(descending=True)
        idx = batch_probs.argsort(descending=True)
        batch_probs = batch_probs[:num_proposals]
        topk_idx = idx[:num_proposals].flatten()
        batch_proposals = torch.gather(batch_proposals,topk_idx)
        #nmss(tt.tensor(batch_proposals.numpy()), tt.tensor(batch_probs.numpy()), nms_threshold).shape
        # For each image, run a total-level NMS, and choose topk results.
        keep = nms(bboxes=batch_proposals.unsqueeze(axis=0), scores=batch_probs.unsqueeze(axis=0).unsqueeze(axis=0), score_threshold=nms_threshold,
                              nms_top_k=post_nms_top_n,
                              keep_top_k=post_nms_top_n,normalized=False)
        #keep = keep[:post_nms_top_n]
        batch_proposals =keep[:,2:] #batch_proposals[keep]
        #batch_probs = batch_probs[keep]
        # cons the rois
        batch_inds = torch.ones((batch_proposals.shape[0], 1)).cast('float32') * bid
        batch_rois = cat([batch_inds, batch_proposals], axis=1)
        return_rois.append(batch_rois)

    if batch_per_gpu == 1:
        return batch_rois
    else:
        concated_rois = cat(return_rois, axis=0)
        return concated_rois
