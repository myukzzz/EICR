# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list


#from denseclip.untils import tokenize
#DenseCLIP





#from ..backbone import build_backbone
from ..backbone import build_backbone_1
#from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
import torch.nn.functional as F

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.spacial_dim = spacial_dim

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        cls_pos = self.positional_embedding[0:1, :]
        # spatial_pos = F.interpolate(self.positional_embedding[1:,].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = self.positional_embedding[1:].reshape(self.spacial_dim, self.spacial_dim, self.embed_dim)[:H, :W]
        spatial_pos = spatial_pos.reshape(-1, self.embed_dim)
        # spatial_pos = spatial_pos.reshape(self.embed_dim, H*W).permute(1, 0)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)

        x = x + positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        x = x.permute(1, 2, 0)
        global_feat = x[:, :, 0]
        feature_map = x[:, :, 1:].reshape(B, -1, H, W)
        return global_feat, feature_map

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg.clone()
        self.backbone = build_backbone_1(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        use_CLIP=False
        self.use_CLIP=use_CLIP











        #############################################################################

        #DenseCLIP
        if use_CLIP:
            self.proj_CLIP = nn.Linear(2048 + 151, 2048)
            self.attnpool = AttentionPool2d(1344 // 32, 2048, 32, 512)
            #self.attnpool_fpn = AttentionPool2d(1344 // 32, 256, 32, 512)
            self.context_length=5
            self.token_embed_dim=512


            f = open("VGclasses.txt", "r")
            class_names = []
            class_list = f.readlines()
            for classes in class_list:
                classes = classes.strip('\n')
                class_names.append(classes)
            self.class_names=class_names
            self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])

            text_encoder = {'type': 'CLIPTextContextEncoder', 'context_length': 13, 'embed_dim': 512, 'transformer_width': 512, 'transformer_heads': 8, 'transformer_layers': 12, 'style': 'pytorch', 'pretrained': 'pretrained/RN101.pt'}
            context_decoder = {'type': 'ContextDecoder', 'transformer_width': 256, 'transformer_heads': 4, 'transformer_layers': 3, 'visual_dim': 512, 'dropout': 0.1, 'style': 'pytorch'}
            # pretrained='pretrained/RN101.pt'
            # assert text_encoder.get('pretrained') is None, \
            #         'both text encoder and segmentor set pretrained weight'
            # text_encoder.pretrained = pretrained
            self.text_encoder = build_backbone(text_encoder)
            self.context_decoder = build_backbone(context_decoder)
            context_length = self.text_encoder.context_length - self.context_length
            self.contexts = nn.Parameter(torch.randn(1, context_length, self.token_embed_dim))
            nn.init.trunc_normal_(self.contexts)
            self.text_dim=512
            self.tau=0.07
            self.gamma = nn.Parameter(torch.ones(self.text_dim) * 1e-4)
            self.text_encoder.init_weights()
        #print("111111111111111111")

        #######################################################################################

    def compute_score_maps(self,feat, x_local,text_features):
        # B, K, C
        visual_embeddings = x_local
        text_features = F.normalize(text_features, dim=-1)
        visual_embeddings = F.normalize(visual_embeddings, dim=1)
        score_map3 = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text_features) / self.tau
        score_map0 = F.upsample(score_map3, feat[0].shape[2:], mode='bilinear')
        score_maps = [score_map0, None, None, score_map3]
        return score_maps

    def compute_text_features(self, x,x_global, x_local, dummy=False):
        """compute text features to each of x
        Args:
            x ([list]): list of features from the backbone,
                x[4] is the output of attentionpool2d
        """
        global_feat, visual_embeddings = x_global, x_local

        B, C, H, W = visual_embeddings.shape

        visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        # text embeddings is (B, K, C)
        if dummy:
            text_embeddings = torch.randn(B, len(self.texts), C, device=global_feat.device)
        else:
            text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff

        return text_embeddings

    def extract_feat(self,feat, x_global, x_local,use_seg_loss=False, dummy=False):
        """Directly extract features from the backbone+neck."""

        text_features = self.compute_text_features(feat,x_global, x_local, dummy=dummy)
        score_maps = self.compute_score_maps(feat,x_local, text_features)
        x=feat
        # x = list(features[:-1])
        B,C,H,W=x[3].shape
        x=list(x)
        x[3] = torch.cat([x[3], score_maps[3]], dim=1)
        x_out=x[3].reshape(B, -1, H*W).permute(2, 0, 1)#HW,2,C
        x_out = self.proj_CLIP(x_out).permute(1, 2, 0)
        x[3]=x_out.reshape(B, -1, H,W)
        x = tuple(x)

        # if self.with_neck:
        #     x = self.neck(x)

        if use_seg_loss:
            return x, score_maps[0]
        else:
            return x
    def compute_seg_loss(self, img,score_map,gt_bboxes,gt_labels):
        #print("gt_bboxes",gt_bboxes)
        #print("gt_labels", gt_labels)
        target, mask = self.build_seg_target(img,gt_bboxes,gt_labels)
        loss = F.binary_cross_entropy(F.sigmoid(score_map), target, weight=mask, reduction='sum')
        loss = loss / mask.sum()
        loss = {'loss_aux_seg': loss}
        return loss

    def build_seg_target(self, img, gt_bboxes, gt_labels):
        B, C, H, W = img.shape
        H //= 4
        W //= 4
        target = torch.zeros(B, len(self.class_names), H, W)
        mask = torch.zeros(B, 1, H, W)
        for i, (bboxes, gt_labels) in enumerate(zip(gt_bboxes, gt_labels)):
            bboxes = (bboxes / 4).long()
            bboxes[:, 0] = bboxes[:, 0].clamp(0, W - 1)
            bboxes[:, 1] = bboxes[:, 1].clamp(0, H - 1)
            bboxes[:, 2] = bboxes[:, 2].clamp(0, W - 1)
            bboxes[:, 3] = bboxes[:, 3].clamp(0, H - 1)
            for bbox, label in zip(bboxes, gt_labels):
                target[i, label, bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1
                mask[i, :, bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1
        mask = mask.expand(-1, len(self.class_names), -1, -1)
        target = target.to(img.device)
        mask = mask.to(img.device)
        return target, mask


    def forward(self, images, targets=None, logger=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)#B*H*W





        ##############################froze####################################
        #DensCLIP
        if self.use_CLIP:
            feat=self.backbone.body(images.tensors)
            x_global, x_local = self.attnpool(feat[3])
            feat,score_map = self.extract_feat(feat, x_global, x_local,use_seg_loss=True)
            features=self.backbone.fpn(feat)
            gt_bboxes=[]
            gt_labels=[]
            for tags in range(len(targets)):
                gt_bboxes.append(targets[tags].bbox)
                gt_labels.append(targets[tags].extra_fields['labels'])
        else:
            features = self.backbone(images.tensors)  # 5*B*H*W
        #print("11111111111111111111")
        ######################################################################






        #print("1111111111111111111")
        ###################froze####################
        proposals, proposal_losses = self.rpn(images, features, targets)

        # x_global, x_local = self.attnpool(features[3])
        # feat,score_map,score_map_local = self.extract_feat(features, x_global, x_local,use_seg_loss=True)
        #print("roihead:",self.roi_heads)#ROIhead
        #targets:B*num_of_box*4
        #targets_extra_fields:B*num_of_box^2
        if self.roi_heads:##############################################
            #detector_losses={}
            #x, detections, loss_box = self.roi_heads.box(features, proposals, targets)
            #x, result, relation_losses = self.roi_heads.relation(features, detections, targets, logger)
            #x, result, detector_losses = self.roi_heads(features,score_map_local,proposals,targets,logger)

            x, result, detector_losses = self.roi_heads(features, proposals, targets, logger)
            #detector_losses.update(relation_losses)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}
        #print("self.training:", self.training)
        if self.training:
            losses = {}
            losses.update(detector_losses)




            #################################################
            # DensCLIP
            if self.use_CLIP:
                self.segloss=self.compute_seg_loss(images.tensors,score_map, gt_bboxes, gt_labels)
                losses.update(self.segloss)

            ############################################################################



            if not self.cfg.MODEL.RELATION_ON:
                # During the relationship training stage, the rpn_head should be fixed, and no loss. 
                losses.update(proposal_losses)
            return losses

        return result
