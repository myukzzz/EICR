'''Rectified Identity Cell'''

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_co_attention import Self_Attention_Encoder, Cross_Attention_Encoder
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import obj_edge_vectors,\
    to_onehot, nms_overlaps, encode_box_info



#DenseCLIP
#from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
#from denseclip.untils import tokenize

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        #self.positional_embedding = nn.Parameter(torch.randn(spacial_dim+1, embed_dim) / embed_dim ** 0.5)
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.spacial_dim = spacial_dim

    def forward(self, x):
        B, N, C  = x.shape
        x = x.permute(1, 0, 2)  # NCHW -> (HW)NC
        #x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        #cls_pos = self.positional_embedding[0:1, :]
        # spatial_pos = F.interpolate(self.positional_embedding[1:,].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        #spatial_pos = self.positional_embedding[1:].reshape(self.spacial_dim, self.embed_dim)[:N]
        spatial_pos = self.positional_embedding.reshape(self.spacial_dim, self.embed_dim)[:N]
        spatial_pos = spatial_pos.reshape(-1, self.embed_dim)
        # spatial_pos = spatial_pos.reshape(self.embed_dim, H*W).permute(1, 0)
        #positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)

        positional_embedding = spatial_pos

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

        feature_map = x.permute(1, 2, 0)
        #global_feat = x[:, :, 0]
        #feature_map = x[:, :, :].reshape(B, -1, N)
        #return global_feat, feature_map
        return  feature_map

class Self_Attention_Cell(nn.Module):
    def __init__(self, config, hidden_dim=None):
        super(Self_Attention_Cell, self).__init__()
        self.cfg = config
        if hidden_dim is None:
            self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        else:
            self.hidden_dim = hidden_dim
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.SA_transformer_encoder = Self_Attention_Encoder(self.num_head, self.k_dim,
                                              self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)


    def forward(self, x, textual_feats=None, num_objs=None):
        assert num_objs is not None
        outp = self.SA_transformer_encoder(x, num_objs)

        return outp

class Cross_Attention_Cell(nn.Module):
    def __init__(self, config, hidden_dim=None):
        super(Cross_Attention_Cell, self).__init__()
        self.cfg = config
        if hidden_dim is None:
            self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        else:
            self.hidden_dim = hidden_dim
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.CA_transformer_encoder = Cross_Attention_Encoder(self.num_head, self.k_dim,
                                self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)


    def forward(self, x, textual_feats, num_objs=None):
        assert num_objs is not None
        outp = self.CA_transformer_encoder(x, textual_feats, num_objs)

        return outp

class Single_Layer_Hybrid_Attention(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, config):
        super().__init__()
        self.SA_Cell_vis = Self_Attention_Cell(config)
        self.SA_Cell_txt = Self_Attention_Cell(config)
        self.CA_Cell_vis = Cross_Attention_Cell(config)
        self.CA_Cell_txt = Cross_Attention_Cell(config)

    def forward(self, visual_feats, text_feats, num_objs):
        tsa = self.SA_Cell_txt(text_feats, num_objs=num_objs)
        tca = self.CA_Cell_txt(text_feats, visual_feats, num_objs=num_objs)
        vsa = self.SA_Cell_vis(visual_feats, num_objs=num_objs)
        vca = self.CA_Cell_vis(visual_feats, text_feats, num_objs=num_objs)
        textual_output = tsa + tca
        visual_output = vsa + vca

        return visual_output, textual_output

class SHA_Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, config, n_layers):
        super().__init__()
        self.cfg = config
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM
        self.cross_module = nn.ModuleList([
            Single_Layer_Hybrid_Attention(config)
            for _ in range(n_layers)])

    def forward(self, visual_feats, text_feats, num_objs):
        visual_output = visual_feats
        textual_output = text_feats

        for enc_layer in self.cross_module:
            visual_output, textual_output = enc_layer(visual_output, textual_output, num_objs)

        visual_output = visual_output + textual_output

        return visual_output, textual_output

class SHA_Context(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super().__init__()
        self.cfg = config
        # setting parameters
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        self.in_channels = in_channels
        self.obj_dim = in_channels
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER
        self.edge_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER

        # the following word embedding layer should be initalize by glove.6B before using
        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(embed_vecs, non_blocking=True)

        # position embedding
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])
        self.lin_obj_visual = nn.Linear(self.in_channels + 128, self.hidden_dim)
        self.lin_obj_textual = nn.Linear(self.embed_dim, self.hidden_dim)

        self.lin_edge_visual = nn.Linear(self.hidden_dim + self.in_channels, self.hidden_dim)
        self.lin_edge_textual = nn.Linear(self.embed_dim, self.hidden_dim)

        self.out_obj = nn.Linear(self.hidden_dim, self.num_obj_cls)

        self.context_obj = SHA_Encoder(config, self.obj_layer)
        self.context_edge = SHA_Encoder(config, self.edge_layer)

        use_CLIP = False
        self.use_CLIP = use_CLIP
        # DenseCLIP
        if use_CLIP:
            self.proj_CLIP = nn.Linear(512 + 151, 512)
            self.attnpool = AttentionPool2d(80, 512, 32, 512)
            # self.attnpool_fpn = AttentionPool2d(1344 // 32, 256, 32, 512)
            self.context_length = 5
            self.token_embed_dim = 512

            f = open("VGclasses.txt", "r")
            class_names = []
            class_list = f.readlines()
            for classes in class_list:
                classes = classes.strip('\n')
                class_names.append(classes)
            self.class_names = class_names
            self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])

            text_encoder = {'type': 'CLIPTextContextEncoder', 'context_length': 13, 'embed_dim': 512,
                            'transformer_width': 512, 'transformer_heads': 8, 'transformer_layers': 12,
                            'style': 'pytorch', 'pretrained': 'pretrained/RN101.pt'}
            context_decoder = {'type': 'ContextDecoder', 'transformer_width': 256, 'transformer_heads': 4,
                               'transformer_layers': 3, 'visual_dim': 512, 'dropout': 0.1, 'style': 'pytorch'}
            # pretrained='pretrained/RN101.pt'
            # assert text_encoder.get('pretrained') is None, \
            #         'both text encoder and segmentor set pretrained weight'
            # text_encoder.pretrained = pretrained
            self.text_encoder = build_backbone(text_encoder)
            self.context_decoder = build_backbone(context_decoder)
            context_length = self.text_encoder.context_length - self.context_length
            self.contexts = nn.Parameter(torch.randn(1, context_length, self.token_embed_dim))
            nn.init.trunc_normal_(self.contexts)
            self.text_dim = 512
            self.tau = 0.07
            self.gamma = nn.Parameter(torch.ones(self.text_dim) * 1e-4)
            self.text_encoder.init_weights()

    def compute_score_maps(self, feat, x_local, text_features):
        # B, K, C
        visual_embeddings = x_local
        text_features = F.normalize(text_features, dim=-1)
        visual_embeddings = F.normalize(visual_embeddings, dim=1)
        score_map = torch.einsum('bcn,bkc->bkn', visual_embeddings, text_features) / self.tau
        #score_map0 = F.upsample(score_map3, feat[0].shape[2:], mode='bilinear')
        #score_maps = [score_map0, None, None, score_map3]
        return score_map

    #def compute_text_features(self, x, x_global, x_local, dummy=False):
    def compute_text_features(self, x, x_local, dummy=False):
        """compute text features to each of x
        Args:
            x ([list]): list of features from the backbone,
                x[4] is the output of attentionpool2d
        """
        # global_feat, visual_embeddings =  x_global,x_local
        #
        # B, C, N = visual_embeddings.shape
        #
        # visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, N)],
        #                            dim=2).permute(0, 2, 1)  # B, N, C
        #
        # # text embeddings is (B, K, C)
        # if dummy:
        #     text_embeddings = torch.randn(B, len(self.texts), C, device=global_feat.device)
        # else:
        #     text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        # text_diff = self.context_decoder(text_embeddings, visual_context)
        # text_embeddings = text_embeddings + self.gamma * text_diff


        visual_embeddings =  x_local
        #global_feat=visual_embeddings.mean(dim=2, keepdim=True).unsqueeze(2)
        B, C, N = visual_embeddings.shape

        visual_context = visual_embeddings.permute(0, 2, 1)
        # visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, N)],
        #                            dim=2).permute(0, 2, 1)

        # text embeddings is (B, K, C)
        if dummy:
            text_embeddings = torch.randn(B, len(self.texts), C, device=visual_embeddings.device)
        else:
            text_embeddings = self.text_encoder(self.texts.to(visual_embeddings.device), self.contexts).expand(B, -1, -1)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff

        return text_embeddings

    #def extract_feat(self, feat, x_global, x_local, use_seg_loss=False, dummy=False):
    def extract_feat(self, feat, x_local, use_seg_loss=False, dummy=False):
        """Directly extract features from the backbone+neck."""

        # text_features = self.compute_text_features(feat, x_global, x_local, dummy=dummy)
        # score_maps = self.compute_score_maps(feat, x_local, text_features)
        # x = feat.permute(0, 2, 1)
        # # x = list(features[:-1])
        # x = torch.cat([x, score_maps], dim=1)
        # x_out = x.permute(0, 2, 1)
        # x = self.proj_CLIP(x_out).squeeze(0)
        # score_maps=score_maps.permute(0, 2, 1)

        text_features = self.compute_text_features(feat, x_local, dummy=dummy)
        score_maps = self.compute_score_maps(feat, x_local, text_features)
        x = feat.permute(0, 2, 1)
        # x = list(features[:-1])
        x = torch.cat([x, score_maps], dim=1)
        x_out = x.permute(0, 2, 1)
        x = self.proj_CLIP(x_out).squeeze(0)
        score_maps=score_maps.permute(0, 2, 1)




        # if self.with_neck:
        #     x = self.neck(x)

        if use_seg_loss:
            return x, score_maps[0]
        else:
            return x

    def forward(self, roi_features, proposals, logger=None):
        # labels will be used in DecoderRNN during training
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        # bbox embedding will be used as input
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.bbox_embed(encode_box_info(proposals))

        # encode objects with transformer

        num_objs = [len(p) for p in proposals]







        obj_pre_rep_vis = cat((roi_features, pos_embed), -1)
        obj_pre_rep_vis = self.lin_obj_visual(obj_pre_rep_vis)
        obj_pre_rep_txt = obj_embed
        obj_pre_rep_txt = self.lin_obj_textual(obj_pre_rep_txt)
        obj_feats_vis, _, = self.context_obj(obj_pre_rep_vis, obj_pre_rep_txt, num_objs)
        obj_feats = obj_feats_vis

        #DenseCLIPvision
        if self.use_CLIP:
            new_obj_feats=[]
            scoremaps= []
            CLIP_obj_feats = list(obj_feats.split(num_objs, dim=0))
            for obj_feat in CLIP_obj_feats:
                if len(obj_feat) == 0:
                    print("#############################object_feat==0########################################")
                    continue
                obj_feat=obj_feat.unsqueeze(0)
                #x_global, x_local = self.attnpool(obj_feat)
                x_local = self.attnpool(obj_feat)
                #obj_feat, score_map = self.extract_feat(obj_feat, x_global, x_local, use_seg_loss=True)
                obj_feat, score_map = self.extract_feat(obj_feat, x_local, use_seg_loss=True)
                new_obj_feats.append(obj_feat)
                scoremaps.append(score_map)
            obj_feats=cat(new_obj_feats,dim=0)
            scoremaps = cat(scoremaps, dim=0)





        # predict obj_dists and obj_preds
        if self.mode == 'predcls':
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
            edge_pre_rep_vis = cat((roi_features, obj_feats), dim=-1)
            edge_pre_rep_txt = self.obj_embed2(obj_labels)
        else:
            obj_dists = self.out_obj(obj_feats)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1
            edge_pre_rep_vis = cat((roi_features, obj_feats), dim=-1)
            edge_pre_rep_txt = self.obj_embed2(obj_preds)

        # edge context
        edge_pre_rep_vis = self.lin_edge_visual(edge_pre_rep_vis)
        edge_pre_rep_txt = self.lin_edge_textual(edge_pre_rep_txt)
        edge_ctx_vis, _ = self.context_edge(edge_pre_rep_vis, edge_pre_rep_txt, num_objs)
        edge_ctx = edge_ctx_vis
        if self.use_CLIP:
            return obj_dists, obj_preds, edge_ctx,scoremaps
        else:
            return obj_dists, obj_preds, edge_ctx

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds



if __name__ == '__main__':
    pass


