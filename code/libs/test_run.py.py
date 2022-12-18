cls_logits = torch.cat(cls_logits, dim=1)
reg_outputs = torch.cat(reg_outputs, dim=1)
ctr_logits = torch.cat(ctr_logits, dim=1)
all_cls_labels = []
all_box_targets = []
all_center_dists = []
all_points = torch.cat([level_points.reshape(-1, 2) for level_points in points], dim=0)
point_strides = torch.cat([
    torch.ones_like(level_points.reshape(-1, 2)[:, 0]) * strides[level_idx]
    for level_idx, level_points in enumerate(points)
], dim=0)
point_reg_ranges = torch.cat([
    torch.ones_like(level_points.reshape(-1, 2)[:, 0])[:, None] * reg_range[level_idx, None]
    for level_idx, level_points in enumerate(points)
], dim=0)
for target_idx, target in enumerate(targets): #This loops through the batch
    target_areas = target['area']
    target_boxes = target['boxes']
    target_lefts, target_tops, target_rights, target_bottoms = target_boxes.unsqueeze(0).unbind(dim=2)
    target_centers = (target_boxes[:, :2] + target_boxes[:, 2:])/2
    # Ensure point is within a sub-box
    center_dists = (all_points[:, None, :] - target_centers[None, :, :]).abs() #Distance from all points in this level to each center
    all_center_dists.append(center_dists)
    max_center_offsets = (
        center_dists
        .max(dim=-1) #only care about larger distance (dim=-1 chooses between x and y distance)
        .values #max returns both larger value and indices
    ) #This is now in the form array of max distances to each bounding box
    min_center_offsets = center_dists.min(dim=-1).values
    points_in_radius = max_center_offsets <= center_sampling_radius * point_strides[:, None]
    #Ensure point is within a bounding box (sub_box for later stages may be larger than bounding boxes)
    point_xs = all_points[:, 0].unsqueeze(1) #Unsqueeze to make shape compatible with targets
    point_ys = all_points[:, 1].unsqueeze(1)
    points_in_bbs = (
        torch.stack([ # Tensor of shape [Nx*Ny, Ntargets, , 4]
            point_xs - target_lefts, #distance to left sides (> 0 means right of )
            point_ys - target_tops, #distance to tops (> 0 means below it)
            target_rights - point_xs, #distance to right sides (> 0 means left of it)
            target_bottoms - point_ys, #distance to bottoms (> 0 means above it)
        ], dim=-1) #Stacking on the last dimension makes each innermost array [l, t, r, b]
        .min(dim=-1) #We only care about the lowest distance, since any < 0 disqualifies the point
        .values > 0) #If any of the above distances is < 0, the point is disqualified
    #Ensure points in regression range
    points_in_reg_range = (
        (min_center_offsets > point_reg_ranges[:, 0, None]) & 
        (max_center_offsets < point_reg_ranges[:, 1, None])
    )
    all_pos_points = points_in_radius & points_in_bbs & points_in_reg_range
    #For regression and centeredness targets, 
    # If points match multiple boxes, take the box with the smallest area
    # Subtract from large number and take max to find minimum, since doing it straight will find the 0s
    min_areas, min_idxs = (all_pos_points * (1e8 - target_areas[None, :])).max(dim=1)
    min_idxs[min_areas < 1e-5] = -1 #Unmatched points should not reference an index in the target box array
    #We now have either -1 for each point if should not be assigned to a target,
    # Or the index of the target
    point_cls_labels = target["labels"][min_idxs]
    point_cls_labels[min_idxs < 0] = -1 #This allows finding FG mask without storing another array
    point_box_targets = target_boxes[min_idxs] #Don't need to set now since we will use FG mask later
    all_cls_labels.append(point_cls_labels.reshape(-1, 1))
    all_box_targets.append(point_box_targets.reshape(-1, 4))

all_cls_labels = torch.stack(all_cls_labels).squeeze(-1) #[bs, nx*ny]
all_box_targets = torch.stack(all_box_targets) #[bs, nx*ny, 4]
#all_center_dists = torch.stack(all_center_dists)

fg_mask = all_cls_labels >= 0
n_fg = fg_mask.sum().item()

#classification loss
cls_targets = torch.zeros_like(cls_logits)
cls_targets[fg_mask, all_cls_labels[fg_mask]] = 1.0 #Sets the appropriate logits to 1, fg_mask is 2d
cls_loss = sigmoid_focal_loss(cls_logits, cls_targets, reduction="sum")

#regression loss
#Convert regression outputs from distances to coordinates for giou_loss
box_target_centers = (all_box_targets[:, :, :2] + all_box_targets[:, :, 2:])/2
center_xs, center_ys = box_target_centers.unbind(dim=-1)
reg_output_strided = reg_outputs * point_strides[None, :, None]
reg_lefts = center_xs - reg_output_strided[:, :, 0]
reg_tops = center_ys - reg_output_strided[:, :, 1]
reg_rights = center_xs + reg_output_strided[:, :, 2]
reg_bottoms = center_ys + reg_output_strided[:, :, 3]
reg_coords = torch.stack([reg_lefts, reg_tops, reg_rights, reg_bottoms], dim=-1)

reg_outputs_fg = reg_coords[fg_mask] # [n_fg, 4]
box_targets_fg = all_box_targets[fg_mask] # [n_fg, 4]
reg_loss = giou_loss(reg_outputs_fg, box_targets_fg, reduction="sum")

#centeredness loss
point_xs = point_xs.unsqueeze(0).squeeze(-1)
point_ys = point_ys.unsqueeze(0).squeeze(-1)
point_left_offsets = (point_xs - all_box_targets[..., 0]).abs()
point_top_offsets = (point_ys - all_box_targets[..., 1]).abs()
point_right_offsets = (point_xs - all_box_targets[..., 2]).abs()
point_bottom_offsets = (point_ys - all_box_targets[..., 3]).abs()
horizontal_offsets = torch.stack([point_left_offsets, point_right_offsets],
    dim=-1)
vertical_offsets = torch.stack([point_top_offsets, point_bottom_offsets],
    dim=-1)

#calculate center score
point_ctr_scores = torch.sqrt(
    horizontal_offsets.min(dim=-1).values/horizontal_offsets.max(dim=-1).values *
    vertical_offsets.min(dim=-1).values/vertical_offsets.max(dim=-1).values
)

ctr_loss = nn.functional.binary_cross_entropy_with_logits(
    ctr_logits.squeeze(-1)[fg_mask], # [bs, nx*ny]
    point_ctr_scores[fg_mask],
    reduction="sum"
)

final_loss = cls_loss + reg_loss + ctr_loss

n_fg_norm = max(1, n_fg)
losses = {
    "cls_loss": cls_loss/n_fg_norm,
    "reg_loss": reg_loss/n_fg_norm,
    "ctr_loss": ctr_loss/n_fg_norm,
    "final_loss": final_loss/n_fg_norm
}


i, image_shape = 0, image_shapes[0]
# level_idx = 0
# level_points = points[level_idx].reshape([-1, 2])
curr_cls_logits = [layer_logits[i] for layer_logits in cls_logits]
curr_reg_outputs = [layer_regs[i] for layer_regs in reg_outputs]
curr_ctr_logits = [layer_ctr[i] for layer_ctr in ctr_logits]

all_boxes = []
all_scores = []
all_labels = []

#level_idx, level_points = 0, points[0].reshape([-1, 2])
for level_idx, level_points in enumerate(points):
    level_points = level_points.reshape([-1, 2])
    level_cls = curr_cls_logits[level_idx]
    level_reg = curr_reg_outputs[level_idx] * strides[level_idx]
    level_ctr = curr_ctr_logits[level_idx]
    object_scores = torch.sqrt(torch.sigmoid(level_cls) * torch.sigmoid(level_ctr))
    #"Decode" means convert from predicted distance from each point to the side of the bounding box
    # to the actual point coordinates of the box this point is supposed to be in
    box_lefts = level_points[:, 0] - level_reg[:, 0]
    box_tops = level_points[:, 1] - level_reg[:, 1]
    box_rights = level_points[:, 0] + level_reg[:, 2]
    box_bottoms = level_points[:, 1] + level_reg[:, 3]
    #Coordinates of predicted boxes for each point
    box_points = torch.stack([box_lefts, box_tops, box_rights, box_bottoms], dim=-1)
    keep_mask = object_scores > score_thresh
    keep_idxs, keep_classes = torch.where(keep_mask)
    kept_boxes = box_points[keep_idxs]
    all_boxes.append(kept_boxes)
    all_scores.append(object_scores[keep_mask])
    all_labels.append(keep_classes)

#b) combine object candidates
all_boxes = torch.cat(all_boxes, dim=0)
all_scores = torch.cat(all_scores, dim=0)
all_labels = torch.cat(all_labels, dim=0)

#b) keep top k candidates
top_scores, top_idxs = all_scores.topk(k=min(topk_candidates, all_scores.shape[0]))
top_boxes = all_boxes[top_idxs]
top_labels = all_labels[top_idxs]

#c) Remove boxes outside image boundaries
outside_boxes = (
    (top_boxes[:, 0] < 0) | 
    (top_boxes[:, 1] < 0) | 
    (top_boxes[:, 2] > image_shape[0]) | 
    (top_boxes[:, 3] > image_shape[1])
)
top_boxes = top_boxes[~outside_boxes]
top_scores = top_scores[~outside_boxes]
top_labels = top_labels[~outside_boxes]

#d) run NMS to remove any duplicated boxes
nms_keeps = batched_nms(top_boxes, top_scores, top_labels, nms_thresh)
nms_boxes = top_boxes[nms_keeps]
nms_scores = top_scores[nms_keeps]
nms_labels = top_labels[nms_keeps]

#e) keep the top-k boxes after NMS
final_scores, final_idxs = nms_scores.topk(k=min(detections_per_img, nms_scores.shape[0]))
final_boxes = nms_boxes[final_idxs]
final_labels = nms_labels[final_idxs]

output.append({
    "boxes": final_boxes,
    "scores": final_scores,
    "labels": final_labels
})