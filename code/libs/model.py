import math
import torch
import torchvision

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.ops.boxes import batched_nms

import torch
from torch import nn

# point generator
from .point_generator import PointGenerator

# input / output transforms
from .transforms import GeneralizedRCNNTransform

# loss functions
from .losses import sigmoid_focal_loss, giou_loss

# import pickle
# with open('objs.pkl', 'rb') as f:
#    targets, center_sampling_radius, strides, points, reg_range, cls_logits, reg_outputs, ctr_logits = pickle.load(f)
        
# import pickle
# with open('inf.pkl', 'rb') as f:
#     score_thresh, nms_thresh, detections_per_img, topk_candidates, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes = pickle.load(f)


class FCOSClassificationHead(nn.Module):
    """
    A classification head for FCOS with convolutions and group norms

    Args:
        in_channels (int): number of channels of the input feature.
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 2.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
    """

    def __init__(self, in_channels, num_classes, num_convs=2, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # A separate background category is not needed, as later we will consider
        # C binary classfication problems here (using sigmoid focal loss)
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # see Sec 3.3 in "Focal Loss for Dense Object Detection'
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

    def forward(self, x):
        """
        Fill in the missing code here. The head will be applied to all levels
        of the feature pyramid, and predict a single logit for each location on
        every feature location.

        Without pertumation, the results will be a list of tensors in increasing
        depth order, i.e., output[0] will be the feature map with highest resolution
        and output[-1] will be the feature map with lowest resolution. The list length is
        equal to the number of pyramid levels. Each tensor in the list will be
        of size N x 1 x H x W, storing the classification logits (scores).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        # x is list with [output of FPN layer]
        # python ./train.py ./configs/voc_fcos.yaml
        output = []
        for fpn_output in x:
            layer_logits = self.cls_logits(self.conv(fpn_output))
            bs, _, feat_h, feat_w = layer_logits.shape
            layer_logits = layer_logits.view(bs, -1, self.num_classes, feat_h, feat_w)
            layer_logits = layer_logits.permute(0, 3, 4, 1, 2)
            layer_logits = layer_logits.reshape(bs, -1, self.num_classes)
            output.append(layer_logits)

        #shape: [4, 3000, 20]: [bs, total pixels in all pyramid levels, n_classes]
        return output


class FCOSRegressionHead(nn.Module):
    """
    A regression head for FCOS with convolutions and group norms.
    This head predicts
    (a) the distances from each location (assuming foreground) to a box
    (b) a center-ness score

    Args:
        in_channels (int): number of channels of the input feature.
        num_convs (Optional[int]): number of conv layer. Default: 2.
    """

    def __init__(self, in_channels, num_convs=2):
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # regression outputs must be positive
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Fill in the missing code here. The logic is rather similar to
        FCOSClassificationHead. The key difference is that this head bundles both
        regression outputs and the center-ness scores.

        Without pertumation, the results will be two lists of tensors in increasing
        depth order, corresponding to regression outputs and center-ness scores.
        Again, the list length is equal to the number of pyramid levels.
        Each tensor in the list will of size N x 4 x H x W (regression)
        or N x 1 x H x W (center-ness).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        # x is list with [output of FPN layer]
        # python ./train.py ./configs/voc_fcos.yaml
        reg_outs = []
        cent_scores = []
        for fpn_output in x:
            layer_conv = self.conv(fpn_output)
            reg_output = self.bbox_reg(layer_conv)
            bs, _, feat_h, feat_w = reg_output.shape
            reg_output = reg_output.view(bs, -1, 4, feat_h, feat_w)
            reg_output = reg_output.permute(0, 3, 4, 1, 2)
            reg_output = reg_output.reshape(bs, -1, 4) # [bs, h*w, 4]
            
            cent_score = self.bbox_ctrness(layer_conv)
            bs, _, feat_h, feat_w = cent_score.shape
            cent_score = cent_score.view(bs, -1, 1, feat_h, feat_w) # [bs, 1, 1, h, w]
            cent_score = cent_score.permute(0, 3, 4, 1, 2) # [bs, h, w, 1, 1]
            cent_score = cent_score.reshape(bs, -1, 1) # [bs, h * w, 1]

            reg_outs.append(reg_output)
            cent_scores.append(cent_score)

        #reg_outs = torch.cat(reg_outs, dim=1) #[bs, total h*w, 4]
        #cent_scores = torch.cat(cent_scores, dim=1) #[bs, total h*w, 1]
        return reg_outs, cent_scores


class FCOS(nn.Module):
    """
    Implementation of (simplified) Fully Convolutional One-Stage object detector,
    as desribed in the journal paper: https://arxiv.org/abs/2006.09214

    Args:
        backbone (string): backbone network, only ResNet18 is supported
        backbone_out_feats (List[string]): output feature maps from the backbone network
        backbone_out_feats_dims (List[int]): backbone output features dimensions
        (in increasing depth order)

        fpn_feats_dim (int): output feature dimension from FPN in increasing depth order
        fpn_strides (List[int]): feature stride for each pyramid level in FPN
        num_classes (int): number of output classes of the model (excluding the background)
        regression_range (List[Tuple[int, int]]): box regression range on each level of the pyramid
        in increasing depth order. E.g., [[0, 32], [32 64]] means that the first level
        of FPN (highest feature resolution) will predict boxes with width and height in range of [0, 32],
        and the second level in the range of [32, 64].

        img_min_size (List[int]): minimum sizes of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.

        train_cfg (Dict): dictionary that specifies training configs, including
            center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.

        test_cfg (Dict): dictionary that specifies test configs, including
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
            topk_candidates (int): Number of best detections to keep before NMS.

        * If a new parameter is added in config.py or yaml file, they will need to defined here.
    """

    def __init__(
        self,
        backbone,
        backbone_out_feats,
        backbone_out_feats_dims,
        fpn_feats_dim,
        fpn_strides,
        num_classes,
        regression_range,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        train_cfg,
        test_cfg,
    ):
        super().__init__()
        assert backbone == "ResNet18"
        self.backbone_name = backbone
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.regression_range = regression_range

        return_nodes = {}
        for feat in backbone_out_feats:
            return_nodes.update({feat: feat})

        # backbone network (resnet18)
        self.backbone = create_feature_extractor(
            resnet18(weights=ResNet18_Weights.DEFAULT), return_nodes=return_nodes
        )

        # feature pyramid network (FPN)
        self.fpn = FeaturePyramidNetwork(
            backbone_out_feats_dims,
            out_channels=fpn_feats_dim,
        )

        # point generator will create a set of points on the 2D image plane
        self.point_generator = PointGenerator(
            img_max_size, fpn_strides, regression_range
        )

        # classification and regression head
        self.cls_head = FCOSClassificationHead(fpn_feats_dim, num_classes)
        self.reg_head = FCOSRegressionHead(fpn_feats_dim)

        # image batching, normalization, resizing, and postprocessing
        self.transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, img_mean, img_std
        )

        # other params for training / inference
        self.center_sampling_radius = train_cfg["center_sampling_radius"]
        self.score_thresh = test_cfg["score_thresh"]
        self.nms_thresh = test_cfg["nms_thresh"]
        self.detections_per_img = test_cfg["detections_per_img"]
        self.topk_candidates = test_cfg["topk_candidates"]

    """
    We will overwrite the train function. This allows us to always freeze
    all batchnorm layers in the backbone, as we won't have sufficient samples in
    each mini-batch to aggregate the bachnorm stats.
    """
    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                if hasattr(module, "weight"):
                    module.weight.requires_grad_(False)
                if hasattr(module, "bias"):
                    module.bias.requires_grad_(False)
            else:
                module.train(mode)
        return self

    """
    The behavior of the forward function changes depending if the model is
    in training or evaluation mode.

    During training, the model expects both the input tensors
    (list of tensors within the range of [0, 1]), as well as a targets
    (list of dictionary), containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
          ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses, as well as a final loss as a summation of all three terms.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    See also the comments for compute_loss / inference.
    """

    def forward(self, images, targets):
        # sanity check
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes of shape [N, 4], got {boxes.shape}.",
                    )

        # record the original image size, this is needed to decode the box outputs
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        # the result will be a dictionary {feature name : tensor}
        features = self.backbone(images.tensors)

        # send the features from the backbone into the FPN
        # the result is converted into a list of tensors (list length = #FPN levels)
        # this list stores features in increasing depth order, each of size N x C x H x W
        # (N: batch size, C: feature channel, H, W: height and width)
        fpn_features = self.fpn(features)
        fpn_features = list(fpn_features.values())

        # classification / regression heads
        cls_logits = self.cls_head(fpn_features)
        reg_outputs, ctr_logits = self.reg_head(fpn_features)

        # 2D points (corresponding to feature locations) of shape H x W x 2
        points, strides, reg_range = self.point_generator(fpn_features) #The indices of pixels in the original image that are present in each fpn layer
        # image_shapes = images.image_sizes
        # score_thresh, nms_thresh, detections_per_img, topk_candidates =  self.score_thresh, self.nms_thresh, self.detections_per_img, self.topk_candidates
        # with open('inf.pkl', 'wb') as f:
        #     pickle.dump([score_thresh, nms_thresh, detections_per_img, topk_candidates, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes], f)
        # print("SAVED INF")
        # print("SAVED INF")
        # print("SAVED INF")
        # print("SAVED INF")
        # training / inference
        if self.training:
            # training: generate GT labels, and compute the loss
            losses = self.compute_loss(
                targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
            )
            # return loss during training
            return losses

        else:
            # inference: decode / postprocess the boxes
            detections = self.inference(
                points, strides, cls_logits, reg_outputs, ctr_logits, images.image_sizes
            )
            # rescale the boxes to the input image resolution
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            # return detectrion results during inference
            return detections

    """
    Fill in the missing code here. This is probably the most tricky part
    in this assignment. Here you will need to compute the object label for each point
    within the feature pyramid. If a point lies around the center of a foreground object
    (as controlled by self.center_sampling_radius), its regression and center-ness
    targets will also need to be computed.

    Further, three loss terms will be attached to compare the model outputs to the
    desired targets (that you have computed), including
    (1) classification (using sigmoid focal for all points)
    (2) regression loss (using GIoU and only on foreground points)
    (3) center-ness loss (using binary cross entropy and only on foreground points)

    Some of the implementation details that might not be obvious
    * The output regression targets are divided by the feature stride (Eq 1 in the paper)
    * All losses are normalized by the number of positive points (Eq 2 in the paper)

    The output must be a dictionary including the loss values
    {
        "cls_loss": Tensor (1)
        "reg_loss": Tensor (1)
        "ctr_loss": Tensor (1)
        "final_loss": Tensor (1)
    }
    where the final_loss is a sum of the three losses and will be used for training.
    """

    def compute_loss(
        self, targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
    ):
        # with open('objs.pkl', 'wb') as f:
        #     pickle.dump([targets, center_sampling_radius, strides, points, reg_range, cls_logits, reg_outputs, ctr_logits], f)
        # print("COMPUTE LOS SAVED")
        # print("COMPUTE LOS SAVED")
        # print("COMPUTE LOS SAVED")
        # print("COMPUTE LOS SAVED")
        cls_logits = torch.cat(cls_logits, dim=1)
        reg_outputs = torch.cat(reg_outputs, dim=1)
        ctr_logits = torch.cat(ctr_logits, dim=1)
        # Categorize points as background/foreground
        all_cls_labels = []
        all_box_targets = []
        #all_center_dists = []
        all_points = torch.cat([level_points.reshape(-1, 2) for level_points in points], dim=0)
        point_strides = torch.cat([
            torch.ones_like(level_points.reshape(-1, 2)[:, 0]) * strides[level_idx]
            for level_idx, level_points in enumerate(points)
        ], dim=0)
        point_xs = all_points[:, 0].unsqueeze(1) #Unsqueeze to make shape compatible with targets
        point_ys = all_points[:, 1].unsqueeze(1)
        point_reg_ranges = torch.cat([
            torch.ones_like(level_points.reshape(-1, 2)[:, 0])[:, None] * reg_range[level_idx, None]
            for level_idx, level_points in enumerate(points)
        ], dim=0)
        for target_idx, target in enumerate(targets): #This loops through the batch
            target_areas = target['area']
            target_boxes = target['boxes']
            target_lefts, target_tops, target_rights, target_bottoms = target_boxes.unsqueeze(0).unbind(dim=-1)
            target_centers = (target_boxes[..., :2] + target_boxes[..., 2:])/2
            
            # Ensure point is within a sub-box
            center_dists = (all_points[:, None, :] - target_centers[None, :, :]).abs() #Distance from all points to each center
            #all_center_dists.append(center_dists)
            max_center_offsets = (
                center_dists
                .max(dim=-1) #only care about larger distance (dim=-1 chooses between x and y distance)
                .values #max returns both larger value and indices
            ) #This is now in the form array of max distances to each bounding box
            min_center_offsets = center_dists.min(dim=-1).values

            points_in_radius = max_center_offsets <= self.center_sampling_radius * point_strides[:, None]
            
            #Ensure point is within a bounding box (sub_box for later stages may be larger than bounding boxes)
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
        return losses

    """
    Fill in the missing code here. The inference is also a bit involved. It is
    much easier to think about the inference on a single image
    (a) Loop over every pyramid level
        (1) compute the object scores
        (2) decode the boxes
        (3) only keep boxes with scores larger than self.score_thresh
    (b) Combine all object candidates across levels and keep the top K (self.topk_candidates)
    (c) Remove boxes outside of the image boundaries (due to padding)
    (d) Run non-maximum suppression to remove any duplicated boxes
    (e) keep the top K boxes after NMS (self.detections_per_img)

    Some of the implementation details that might not be obvious
    * As the output regression target is divided by the feature stride during training,
    you will have to multiply the regression outputs by the stride at inference time.
    * Most of the detectors will allow two overlapping boxes from two different categories
    (e.g., one from "shirt", the other from "person"). That means that
        (a) one can decode two same boxes of different categories from one location;
        (b) NMS is only performed within each category.
    * Regression range is not used, as the range is not enforced during inference.
    * image_shapes is needed to remove boxes outside of the images.

    The output must be a list of dictionary items (one for each image) following
    [
        {
            "boxes": Tensor (N x 4)
            "scores": Tensor (N, )
            "labels": Tensor (N, )
        },
    ]
    """

    def inference(
        self, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes
    ):
        score_thresh = self.score_thresh #(float): Score threshold used for postprocessing the detections.
        nms_thresh = self.nms_thresh #(float): NMS threshold used for postprocessing the detections.
        detections_per_img = self.detections_per_img #(int): Number of best detections to keep after NMS.
        topk_candidates = self.topk_candidates #(int): Number of best detections to keep before NMS.
        
        detections = []
        for i, image_shape in enumerate(image_shapes): #Loop through the batch
            #Each level of head outputs is a pyramid layer, with all items in batch at that layer
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

            detections.append({
                "boxes": final_boxes,
                "scores": final_scores,
                "labels": final_labels
            })
        return detections
